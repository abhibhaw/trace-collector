"""Upload one batch's traces to the agent-sim platform, one platform agent per agent_id.

    python3 scripts/upload_traces.py --batch full (--agents id,id | --all) [--dry-run]
        [--url https://demo.peira.tech] [--from-archive] [--limit N]
        [--source langfuse] [--envelope array] [--concurrency 4]

Trace source:
  default         Langfuse's ClickHouse through kubectl (scripts/export_langfuse.py).
  --from-archive  the local Git LFS files data/traces/<batch>/<family>/<agent_id>.jsonl.gz. Generation outputs are
                  converted to OpenAI-style messages, as export_langfuse.py does without --raw-generations.

Credentials: the platform needs a signed-in session. Two ways, never printed or written anywhere:
  - a session cookie (Google sign-in hosts): PLATFORM_SESSION, else --cookie-file (default
    ~/.config/peira/<host>.cookie, mode 0600 or narrower). The file holds the es_session value, or "es_session=<value>".
  - a user name and password: PLATFORM_USERNAME / PLATFORM_PASSWORD, else --creds-file (default
    ~/.config/peira/shimyureta.env, KEY=VALUE lines, mode 0600 or narrower).
The cookie wins when both exist. --url (or PLATFORM_URL) sets the base URL.

Per agent:
  1. Find the platform agent by name (= agent_id) in GET /api/agents; create it only if missing.
  2. Export the agent's traces that are not yet in the state file.
  3. POST /api/agents/{key}/import in chunks of --chunk traces. A host that answers 202 {job} is polled at
     GET /api/jobs/{job}. A chunk counts as done when the result says it stored one run per trace.
     The platform upserts runs by Langfuse trace id, so retries never duplicate.
  4. Record the uploaded trace_ids in the state file, so a re-run uploads only new traces. The state file is
     results/upload-state/<batch>.json for the original host, else results/upload-state/<host>/<batch>.json.
"""
import argparse
import gzip
import http.client
import http.cookiejar
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_langfuse as ex  # noqa: E402

LEGACY_BASE = "https://ejento-shimyureta-v1.devtron.info"
BASE = os.environ.get("PLATFORM_URL", LEGACY_BASE).rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}
COOKIE = "es_session"


class AuthError(RuntimeError):
    pass


DEFAULT_CREDS = os.path.expanduser("~/.config/peira/shimyureta.env")


def host() -> str:
    return urllib.parse.urlparse(BASE).netloc


def _private_file(path):
    """Refuse a credentials file that others can read."""
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise AuthError(f"refusing to read {path}: mode {oct(mode)} is wider than 0600")


def load_cookie(path=None):
    """Return the session cookie value, or None when there is none. Never printed."""
    v = os.environ.get("PLATFORM_SESSION")
    if not v:
        path = os.path.expanduser(path or f"~/.config/peira/{host()}.cookie")
        if not os.path.exists(path):
            return None
        _private_file(path)
        v = open(path).read()
    v = v.strip().strip("'\"")
    if v.startswith(COOKIE + "="):
        v = v[len(COOKIE) + 1:]
    return v.split(";", 1)[0].strip() or None


def load_creds(path=None):
    """Return (username, password): env vars first, else a KEY=VALUE file with mode 0600 or narrower.
    Values are never printed. Error messages name only the file and key names."""
    user, pw = os.environ.get("PLATFORM_USERNAME"), os.environ.get("PLATFORM_PASSWORD")
    if user and pw:
        return user, pw
    path = os.path.expanduser(path or DEFAULT_CREDS)
    if not os.path.exists(path):
        raise AuthError(f"no credentials: set PLATFORM_SESSION, PLATFORM_USERNAME/PLATFORM_PASSWORD or create {path}")
    _private_file(path)
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k.startswith("export "):
                k = k[7:].strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            vals[k] = v
    user, pw = user or vals.get("PLATFORM_USERNAME"), pw or vals.get("PLATFORM_PASSWORD")
    if not user or not pw:
        raise AuthError(f"{path} lacks PLATFORM_USERNAME and/or PLATFORM_PASSWORD")
    return user, pw


class Platform:
    def __init__(self, creds_file=None, cookie_file=None, job_timeout=1800):
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.lock = threading.Lock()
        self.creds_file = creds_file
        self.cookie = load_cookie(cookie_file)
        self.logged_in = self.cookie is not None
        self.job_timeout = job_timeout

    def login(self):
        if self.cookie is not None:
            raise AuthError("the session cookie was refused (HTTP 401): sign in again and replace it")
        user, pw = load_creds(self.creds_file)
        body = json.dumps({"username": user, "password": pw}).encode()
        req = urllib.request.Request(BASE + "/api/auth/login", data=body, method="POST",
                                     headers={"content-type": "application/json"})
        try:
            self.op.open(req, timeout=30).read()
        except urllib.error.HTTPError as e:
            raise AuthError(f"login failed: HTTP {e.code}") from None
        self.logged_in = True

    def call(self, method, path, body=None, headers=None, timeout=60, retries=4):
        """JSON request with login-on-401 and retries on transient errors (for idempotent calls)."""
        for attempt in range(retries + 1):
            h = dict(headers or {})
            if self.cookie is not None:
                h["cookie"] = f"{COOKIE}={self.cookie}"
            req = urllib.request.Request(BASE + path, data=body, method=method, headers=h)
            try:
                with self.lock:
                    if not self.logged_in:
                        self.login()
                with self.op.open(req, timeout=timeout) as r:
                    return json.loads(r.read() or b"null")
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode(errors="replace")
                if e.code == 401 and attempt == 0:
                    with self.lock:
                        self.logged_in = False
                    continue
                if e.code in TRANSIENT and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{method} {path}: HTTP {e.code} {detail}") from None
            # socket.timeout is not a TimeoutError before Python 3.10; a pod restart drops connections mid-reply
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException) as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{method} {path}: {e}") from None

    def agents(self) -> list:
        return self.call("GET", "/api/agents")

    def runs(self, key) -> int:
        for a in self.agents():
            if a.get("key") == key:
                return int(a.get("runs") or 0)
        raise RuntimeError(f"agent key {key} vanished")

    def create(self, name) -> dict:
        # The answer holds the agent's ingest token. Only the key is kept.
        return self.call("POST", "/api/agents", json.dumps({"name": name}).encode(),
                         {"content-type": "application/json"}, retries=0)

    def wait_job(self, job) -> dict:
        """Poll an import job until it ends. -> its result ({stored, ...}). RuntimeError when it failed."""
        t0, pause = time.time(), 1.0
        while True:
            j = self.call("GET", f"/api/jobs/{urllib.parse.quote(job)}")
            if j.get("done"):
                if j.get("status") != "done":
                    raise RuntimeError(f"job {job} {j.get('status')}: {j.get('error')}"[:400])
                return j.get("result") or {}
            if time.time() - t0 > self.job_timeout:
                raise RuntimeError(f"job {job} still {j.get('status')} after {self.job_timeout}s")
            time.sleep(pause)
            pause = min(pause * 1.5, 10)

    def import_file(self, key, path, source) -> dict:
        b = uuid.uuid4().hex
        with open(path, "rb") as f:
            data = f.read()
        fn = os.path.basename(path)
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fn}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
        q = urllib.parse.urlencode({"source": source})
        resp = self.call("POST", f"/api/agents/{urllib.parse.quote(key)}/import?{q}", body,
                         {"content-type": f"multipart/form-data; boundary={b}"}, timeout=600, retries=0)
        if isinstance(resp, dict) and resp.get("job") and "stored" not in resp:
            return self.wait_job(resp["job"])
        return resp


class State:
    def __init__(self, batch):
        sub = () if BASE == LEGACY_BASE else (host(),)
        self.path = os.path.join(ROOT, "results", "upload-state", *sub, f"{batch}.json")
        self.lock = threading.Lock()
        self.d = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def get(self, agent) -> dict:
        return self.d.get(agent, {"key": None, "uploaded": []})

    def record(self, agent, key, ids):
        with self.lock:
            e = self.d.setdefault(agent, {"key": key, "uploaded": []})
            e["key"] = key
            e["uploaded"] = sorted(set(e["uploaded"]) | set(ids))
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.d, f, indent=1, sort_keys=True)
            os.replace(tmp, self.path)


class ClickHouseSource:
    """Traces read from Langfuse's ClickHouse (kubectl exec)."""

    def __init__(self, batch):
        self.batch = batch

    def agents(self) -> list:
        sql = (f"SELECT DISTINCT replaceOne(arrayFirst(x -> startsWith(x, 'agent:'), tags), 'agent:', '') AS a "
               f"FROM events_core FINAL WHERE parent_span_id = '' AND has(tags, 'batch:{self.batch}') ORDER BY a"
               + ex.SETTINGS)
        return [r["a"] for r in ex.ch(sql) if r["a"]]

    def trace_ids(self, agent) -> list:
        return ex.root_trace_ids(self.batch, agent)

    def sizes(self, agent, ids) -> dict:
        return {t: int(v.get("b", 0)) for t, v in ex.trace_stats(ids).items()}

    def export(self, agent, out, envelope, ids) -> list:
        return ex.export(self.batch, agent, out, envelope, ids)["trace_ids"]


class ArchiveSource:
    """Traces read from data/traces/<batch>/<family>/<agent>.jsonl.gz (one trace per line, raw generations)."""

    def __init__(self, batch, limit=None):
        self.dir = os.path.join(ROOT, "data", "traces", batch)
        with open(os.path.join(self.dir, "manifest.json")) as f:
            self.manifest = json.load(f)["agents"]
        self.limit = limit
        self._lines = {}  # agent -> {trace_id: raw line bytes}, one agent at a time per worker
        self.lock = threading.Lock()

    def agents(self) -> list:
        return sorted(self.manifest)

    def _load(self, agent) -> dict:
        with self.lock:
            if agent in self._lines:
                return self._lines[agent]
        m = self.manifest[agent]
        path = os.path.join(self.dir, m["file"])
        with open(path, "rb") as f:
            head = f.read(200)
        if head.startswith(b"version https://git-lfs"):
            raise RuntimeError(f"{path} is a Git LFS pointer: run `git lfs pull`")
        lines = {}
        with gzip.open(path, "rb") as f:
            for line in f:
                if line.strip():
                    tid = json.loads(line)["id"]
                    lines[tid] = line
                    if self.limit and len(lines) >= self.limit:
                        break
        with self.lock:
            self._lines[agent] = lines
        return lines

    def done(self, agent):
        with self.lock:
            self._lines.pop(agent, None)

    def trace_ids(self, agent) -> list:
        return list(self._load(agent))

    def sizes(self, agent, ids) -> dict:
        lines = self._load(agent)
        return {t: len(lines[t]) for t in ids if t in lines}

    def export(self, agent, out, envelope, ids) -> list:
        lines = self._load(agent)
        traces = []
        for t in ids:
            tr = json.loads(lines[t])
            for o in tr.get("observations") or []:
                if o.get("type") == "GENERATION":
                    o["output"] = ex.normalize_generation_output(o.get("output"))
            traces.append(tr)
        body = traces if envelope == "array" else {"data": traces}
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(body, f, ensure_ascii=False)
        return [t["id"] for t in traces]


def do_agent(agent, a, plat, state, keys, create_lock, tmpdir, src) -> dict:
    row = {"agent": agent, "ch": 0, "new": 0, "uploaded": 0, "runs": None, "status": ""}
    try:
        return _do_agent(row, agent, a, plat, state, keys, create_lock, tmpdir, src)
    finally:
        if hasattr(src, "done"):
            src.done(agent)


def _do_agent(row, agent, a, plat, state, keys, create_lock, tmpdir, src) -> dict:
    ch_ids = src.trace_ids(agent)
    row["ch"] = len(ch_ids)
    st = state.get(agent)
    new = [t for t in ch_ids if t not in set(st["uploaded"])]
    row["new"] = len(new)
    key = keys.get(agent)
    if a.dry_run:
        row["runs"] = keys.get(("runs", agent))
        row["status"] = "dry-run" + ("" if key or plat is None else " (would create)")
        return row
    if not new:
        row["runs"] = plat.runs(key) if key else None
        row["status"] = "up to date" if ch_ids else "no traces in source"
        return row
    with create_lock:
        key = keys.get(agent)
        if not key:
            existing = [x for x in plat.agents() if x.get("name") == agent]
            key = existing[0]["key"] if existing else plat.create(agent)["key"]
            keys[agent] = key
    before = plat.runs(key)
    problems, sent = [], 0
    # Chunks are bounded by trace count and by payload bytes (57 MB raw / 14 MB gzipped was accepted in tests).
    sizes = src.sizes(agent, new)
    queue, cur, size = [], [], 0
    for t in new:
        b = int(sizes.get(t, 0))
        if cur and (len(cur) >= a.chunk or size + b > a.chunk_bytes):
            queue.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += b
    if cur:
        queue.append(cur)
    n = 0
    while queue:
        part = queue.pop(0)
        n += 1
        out = os.path.join(tmpdir, f"{agent}.{a.batch}.{n}.json.gz")
        ids = src.export(agent, out, a.envelope, part)
        if not ids:
            continue
        resp, err = None, None
        # The platform upserts runs by Langfuse trace id, so re-sending a file never duplicates runs.
        for attempt in range(a.retries + 1):
            try:
                resp = plat.import_file(key, out, a.source)
                break
            except AuthError:
                raise
            except RuntimeError as e:
                err = e
                if " HTTP 4" in str(e) and " HTTP 408" not in str(e) and " HTTP 429" not in str(e):
                    break
                time.sleep(2 ** attempt * 3)
        os.remove(out)
        stored = ((resp or {}).get("stored") or {}).get("run")
        if stored == len(ids):
            state.record(agent, key, ids)
            sent += stored
        elif len(part) > 1:  # e.g. 413 or a bad trace: split and retry the halves
            h = len(part) // 2
            queue[:0] = [part[:h], part[h:]]
        else:
            problems.append(f"trace {part[0]}: stored {stored} {err or ''}"[:200])
    after = plat.runs(key)
    row["runs"], row["uploaded"] = after, sent
    grew = after - before
    if problems:
        row["status"] = "FAILED " + "; ".join(problems)
    elif grew == sent:
        row["status"] = "ok"
    else:
        # Runs already on the platform (e.g. state file lost) are upserted, not added.
        row["status"] = f"ok (runs +{grew}; {sent - grew} already present, upserted)"
    return row


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--agents")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", help=f"platform base URL (default PLATFORM_URL, else {LEGACY_BASE})")
    ap.add_argument("--from-archive", action="store_true", help="read data/traces/<batch> instead of ClickHouse")
    ap.add_argument("--limit", type=int, help="with --from-archive: at most this many traces per agent")
    ap.add_argument("--source", default="langfuse")
    ap.add_argument("--envelope", choices=["array", "data"], default="array")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=10, help="max traces per import request")
    ap.add_argument("--chunk-bytes", type=int, default=40_000_000, help="max raw payload bytes per import request")
    ap.add_argument("--job-timeout", type=int, default=1800, help="seconds to wait for one import job")
    ap.add_argument("--creds-file", help=f"KEY=VALUE file (default {DEFAULT_CREDS}); env vars win")
    ap.add_argument("--cookie-file", help="file with the es_session cookie (default ~/.config/peira/<host>.cookie)")
    a = ap.parse_args()
    if a.url:
        BASE = a.url.rstrip("/")
    if not ex.SAFE.match(a.batch):
        print("bad batch name", file=sys.stderr)
        return 2
    src = ArchiveSource(a.batch, a.limit) if a.from_archive else ClickHouseSource(a.batch)
    agents = src.agents() if a.all else [x.strip() for x in a.agents.split(",") if x.strip()]
    for x in agents:
        if not ex.SAFE.match(x):
            print(f"bad agent id {x!r}", file=sys.stderr)
            return 2
    conc = max(1, min(4, a.concurrency))
    state = State(a.batch)
    print(f"platform {BASE}, state {os.path.relpath(state.path, ROOT)}", file=sys.stderr)
    keys = {}
    try:
        plat = Platform(a.creds_file, a.cookie_file, a.job_timeout)
        for x in plat.agents():
            if x.get("name") in agents:
                if x["name"] in keys:
                    print(f"WARNING: duplicate platform agents named {x['name']}", file=sys.stderr)
                    continue
                keys[x["name"]] = x["key"]
                keys[("runs", x["name"])] = x.get("runs")
    except AuthError as e:
        if not a.dry_run:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        print(f"note: {e}; dry-run continues without platform data", file=sys.stderr)
        plat = None
    create_lock = threading.Lock()
    rows = []
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(conc) as pool:
        futs = {pool.submit(do_agent, ag, a, plat, state, keys, create_lock, tmp, src): ag for ag in agents}
        for f, ag in futs.items():
            try:
                rows.append(f.result())
            except Exception as e:  # report and keep going
                rows.append({"agent": ag, "ch": "?", "new": "?", "uploaded": 0, "runs": None,
                             "status": f"ERROR {e}"[:200]})
            r = rows[-1]
            print(f"{r['agent']:32} {r['ch']:>6} {r['new']:>5} {r['uploaded']:>6} {str(r['runs']):>6}  {r['status']}",
                  file=sys.stderr, flush=True)
    print(f"{'agent':32} {'traces':>9} {'new':>5} {'uploaded':>8} {'runs':>6}  status")
    for r in sorted(rows, key=lambda r: r["agent"]):
        print(f"{r['agent']:32} {r['ch']:>9} {r['new']:>5} {r['uploaded']:>8} {str(r['runs']):>6}  {r['status']}")
    tot = lambda k: sum(r[k] for r in rows if isinstance(r[k], int))  # noqa: E731
    print(f"\n{len(rows)} agents, {tot('ch')} traces in source, {tot('new')} new, {tot('uploaded')} uploaded")
    bad = [r for r in rows if r["status"].startswith(("FAILED", "ERROR", "MISMATCH"))]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
