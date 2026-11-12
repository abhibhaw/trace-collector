"""Upload one batch's traces from Langfuse's ClickHouse to the agent-sim platform, one platform agent per agent_id.

    python3 scripts/upload_traces.py --batch full (--agents id,id | --all) [--dry-run]
        [--source langfuse] [--envelope array] [--concurrency 4]

Credentials: the platform needs a signed-in session. PLATFORM_USERNAME / PLATFORM_PASSWORD come from the
environment if set, else from --creds-file (default ~/.config/peira/shimyureta.env, KEY=VALUE lines, mode 0600
or narrower). PLATFORM_URL overrides the base URL. Credentials are never printed or written anywhere.

Per agent:
  1. Find the platform agent by name (= agent_id) in GET /api/agents; create it only if missing.
  2. Export the agent's traces that are not yet in the state file (scripts/export_langfuse.py).
  3. POST /api/agents/{key}/import in chunks of --chunk traces; a chunk counts as done when the response says
     it stored one run per trace. The platform upserts runs by Langfuse trace id, so retries never duplicate.
  4. Record the uploaded trace_ids in results/upload-state/<batch>.json, so a re-run uploads only new traces.
"""
import argparse
import http.cookiejar
import json
import os
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

BASE = os.environ.get("PLATFORM_URL", "https://ejento-shimyureta-v1.devtron.info").rstrip("/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class AuthError(RuntimeError):
    pass


DEFAULT_CREDS = os.path.expanduser("~/.config/peira/shimyureta.env")


def load_creds(path=None):
    """Return (username, password): env vars first, else a KEY=VALUE file with mode 0600 or narrower.
    Values are never printed. Error messages name only the file and key names."""
    user, pw = os.environ.get("PLATFORM_USERNAME"), os.environ.get("PLATFORM_PASSWORD")
    if user and pw:
        return user, pw
    path = os.path.expanduser(path or DEFAULT_CREDS)
    if not os.path.exists(path):
        raise AuthError(f"no credentials: set PLATFORM_USERNAME/PLATFORM_PASSWORD or create {path}")
    mode = os.stat(path).st_mode & 0o777
    if mode & 0o077:
        raise AuthError(f"refusing to read {path}: mode {oct(mode)} is wider than 0600")
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
    def __init__(self, creds_file=None):
        self.jar = http.cookiejar.CookieJar()
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.lock = threading.Lock()
        self.logged_in = False
        self.creds_file = creds_file

    def login(self):
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
            req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers or {})
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
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
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
        return self.call("POST", "/api/agents", json.dumps({"name": name}).encode(),
                         {"content-type": "application/json"}, retries=0)

    def import_file(self, key, path, source) -> dict:
        b = uuid.uuid4().hex
        with open(path, "rb") as f:
            data = f.read()
        fn = os.path.basename(path)
        body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fn}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
        q = urllib.parse.urlencode({"source": source})
        return self.call("POST", f"/api/agents/{urllib.parse.quote(key)}/import?{q}", body,
                         {"content-type": f"multipart/form-data; boundary={b}"}, timeout=600, retries=0)


class State:
    def __init__(self, batch):
        self.path = os.path.join(ROOT, "results", "upload-state", f"{batch}.json")
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


def batch_agents(batch) -> list:
    sql = (f"SELECT DISTINCT replaceOne(arrayFirst(x -> startsWith(x, 'agent:'), tags), 'agent:', '') AS a "
           f"FROM events_core FINAL WHERE parent_span_id = '' AND has(tags, 'batch:{batch}') ORDER BY a" + ex.SETTINGS)
    return [r["a"] for r in ex.ch(sql) if r["a"]]


def do_agent(agent, a, plat, state, keys, create_lock, tmpdir) -> dict:
    row = {"agent": agent, "ch": 0, "new": 0, "uploaded": 0, "runs": None, "status": ""}
    ch_ids = ex.root_trace_ids(a.batch, agent)
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
        row["status"] = "up to date" if ch_ids else "no traces in ClickHouse"
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
    sizes = ex.trace_stats(new)
    queue, cur, size = [], [], 0
    for t in new:
        b = int(sizes.get(t, {}).get("b", 0))
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
        s = ex.export(a.batch, agent, out, a.envelope, part)
        if not s["trace_ids"]:
            continue
        resp, err = None, None
        # The platform upserts runs by Langfuse trace id, so re-sending a file never duplicates runs.
        for attempt in range(a.retries + 1):
            try:
                resp = plat.import_file(key, out, a.source)
                break
            except RuntimeError as e:
                err = e
                if " HTTP 4" in str(e) and " HTTP 408" not in str(e) and " HTTP 429" not in str(e):
                    break
                time.sleep(2 ** attempt * 3)
        os.remove(out)
        stored = ((resp or {}).get("stored") or {}).get("run")
        if stored == len(s["trace_ids"]):
            state.record(agent, key, s["trace_ids"])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--agents")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source", default="langfuse")
    ap.add_argument("--envelope", choices=["array", "data"], default="array")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=10, help="max traces per import request")
    ap.add_argument("--chunk-bytes", type=int, default=40_000_000, help="max raw payload bytes per import request")
    ap.add_argument("--creds-file", help=f"KEY=VALUE file (default {DEFAULT_CREDS}); env vars win")
    a = ap.parse_args()
    if not ex.SAFE.match(a.batch):
        print("bad batch name", file=sys.stderr)
        return 2
    agents = batch_agents(a.batch) if a.all else [x.strip() for x in a.agents.split(",") if x.strip()]
    for x in agents:
        if not ex.SAFE.match(x):
            print(f"bad agent id {x!r}", file=sys.stderr)
            return 2
    conc = max(1, min(4, a.concurrency))
    state = State(a.batch)
    plat, keys = Platform(a.creds_file), {}
    try:
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
        futs = {pool.submit(do_agent, ag, a, plat, state, keys, create_lock, tmp): ag for ag in agents}
        for f, ag in futs.items():
            try:
                rows.append(f.result())
            except Exception as e:  # report and keep going
                rows.append({"agent": ag, "ch": "?", "new": "?", "uploaded": 0, "runs": None,
                             "status": f"ERROR {e}"[:200]})
    print(f"{'agent':32} {'ch_traces':>9} {'new':>5} {'uploaded':>8} {'runs':>6}  status")
    for r in sorted(rows, key=lambda r: r["agent"]):
        print(f"{r['agent']:32} {r['ch']:>9} {r['new']:>5} {r['uploaded']:>8} {str(r['runs']):>6}  {r['status']}")
    tot = lambda k: sum(r[k] for r in rows if isinstance(r[k], int))  # noqa: E731
    print(f"\n{len(rows)} agents, {tot('ch')} traces in ClickHouse, {tot('new')} new, {tot('uploaded')} uploaded")
    bad = [r for r in rows if r["status"].startswith(("FAILED", "ERROR", "MISMATCH"))]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
