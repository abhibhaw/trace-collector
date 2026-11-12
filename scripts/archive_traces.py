"""Archive one batch from Langfuse as gzipped JSONL, one file per agent, for git (LFS).

    python3 scripts/archive_traces.py --batch full [--agents a,b] [--jobs 2]

Writes data/traces/<batch>/<family>/<agent_id>.jsonl.gz and data/traces/<batch>/manifest.json.
Each line is one trace in Langfuse public-API shape with its nested observations, exactly as
scripts/export_langfuse.py writes it with --raw-generations (framework-native generation output,
secrets masked, Langfuse resource metadata dropped). Existing files are rebuilt only with --force.
Keep --jobs at 2 or lower: each export reads payload columns from ClickHouse.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import export_langfuse as ex  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--agents", default="")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ex.NORMALIZE = False  # archive the framework-native generation output

    agents = sorted((p.parent.parent.name, p.parent.name) for p in ROOT.glob("agents/*/*/agent.py"))
    if a.agents:
        want = set(a.agents.split(","))
        agents = [x for x in agents if x[1] in want]
    out_root = ROOT / "data" / "traces" / a.batch
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"batch": a.batch, "agents": {}}

    def one(fam_agent):
        fam, agent = fam_agent
        out = out_root / fam / f"{agent}.jsonl.gz"
        if out.exists() and not a.force and agent in manifest["agents"]:
            return agent, manifest["agents"][agent], "kept"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.gz")
        t0 = time.time()
        s = ex.export(a.batch, agent, str(tmp), "jsonl")
        # Verify: line count equals exported traces and the gzip stream is complete.
        with gzip.open(tmp, "rt", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        if lines != s["traces"]:
            raise RuntimeError(f"{agent}: wrote {lines} lines for {s['traces']} traces")
        tmp.replace(out)
        entry = {"family": fam, "file": str(out.relative_to(out_root)), "traces": s["traces"],
                 "observations": s["observations"], "bytes": out.stat().st_size, "sha256": sha256(out)}
        return agent, entry, f"{time.time() - t0:.0f}s"

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, min(a.jobs, 2))) as pool:
        for fut in [pool.submit(one, x) for x in agents]:
            try:
                agent, entry, how = fut.result()
                manifest["agents"][agent] = entry
                print(f"{agent:34} {entry['traces']:>4} traces {entry['observations']:>6} obs "
                      f"{entry['bytes'] / 1e6:7.1f} MB  {how}", flush=True)
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            except Exception as e:  # keep going; report at the end
                failed.append(str(e)[:300])
                print("FAILED", str(e)[:300], flush=True)

    m = manifest["agents"]
    manifest["totals"] = {"agents": len(m), "traces": sum(v["traces"] for v in m.values()),
                          "observations": sum(v["observations"] for v in m.values()),
                          "bytes": sum(v["bytes"] for v in m.values())}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(manifest["totals"]), f"failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
