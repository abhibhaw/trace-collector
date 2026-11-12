"""Run one agent over its task file. One Langfuse trace per task.

    python -m peira_harness.runner --agent agents/<family>/<agent_id> --batch <b> [--limit N] [--concurrency C]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import AgentMeta, llm, tools, tracing

TASK_TIMEOUT_S = int(os.environ.get("PEIRA_TASK_TIMEOUT_S", "600"))


def load_agent(agent_dir: Path):
    path = agent_dir / "agent.py"
    # Make sibling modules in the agent dir and the family dir importable.
    for p in (agent_dir, agent_dir.parent):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(f"agent_{agent_dir.name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses and pydantic resolve forward refs through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    meta = getattr(mod, "AGENT", None)
    if not isinstance(meta, AgentMeta):
        raise TypeError(f"{path}: AGENT must be a peira_harness.AgentMeta")
    if meta.id != agent_dir.name:
        raise ValueError(f"{path}: AGENT.id {meta.id!r} must equal the directory name {agent_dir.name!r}")
    has_run, has_arun = hasattr(mod, "run"), hasattr(mod, "arun")
    if has_run == has_arun:
        raise TypeError(f"{path}: define exactly one of run(task) or arun(task)")
    return mod, meta


def load_tasks(agent_dir: Path, limit: int | None) -> list[dict]:
    tasks = [json.loads(line) for line in (agent_dir / "tasks.jsonl").read_text().splitlines() if line.strip()]
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{agent_dir}: duplicate task ids")
    return tasks[:limit] if limit else tasks


def done_ids(results_file: Path) -> set[str]:
    if not results_file.exists():
        return set()
    out = set()
    for line in results_file.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.add(r["task_id"])
    return out


class Runner:
    def __init__(self, mod, meta: AgentMeta, family: str, batch: str, results_file: Path):
        self.mod, self.meta, self.family, self.batch = mod, meta, family, batch
        self.results_file = results_file
        self._write_lock = threading.Lock()
        self.lf = tracing.client()

    def _record(self, row: dict) -> None:
        with self._write_lock, self.results_file.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _run_id(self, task: dict) -> str:
        return f"{self.meta.id}:{task['id']}:{self.batch}"

    def _attrs(self, task: dict) -> dict:
        m = self.meta
        return dict(
            trace_name=m.id,
            session_id=f"{m.id}:{self.batch}",
            tags=[
                f"framework:{m.framework}",
                f"agent:{m.id}",
                f"family:{self.family}",
                f"complexity:{m.complexity}",
                f"kind:{task.get('kind', 'normal')}",
                f"batch:{self.batch}",
            ],
            metadata={
                "task_id": str(task["id"]),
                "source_url": m.source_url[:200],
                "model": m.model,
                "agent_id": m.id,
            },
        )

    def _finish(self, task: dict, root, trace_id: str, t0: float, output=None, error: BaseException | None = None):
        latency = round(time.time() - t0, 2)
        if error is None:
            root.update(output=output)
            status = "ok"
        else:
            msg = f"{type(error).__name__}: {error}"[:2000]
            root.update(level="ERROR", status_message=msg, output={"error": msg})
            status = "timeout" if isinstance(error, (TimeoutError, asyncio.TimeoutError)) else "error"
        row = {"task_id": task["id"], "trace_id": trace_id, "status": status, "latency_s": latency}
        if output is not None:
            row["output"] = str(output)[:2000]
        if error is not None:
            row["error"] = f"{type(error).__name__}: {error}"[:500]
        self._record(row)
        print(f"[{self.meta.id}] {task['id']} {status} {latency}s trace={trace_id}", flush=True)
        if error is not None and os.environ.get("PEIRA_DEBUG"):
            traceback.print_exception(error)
        return status

    # Sync agents: one thread per in-flight task. A timed-out thread keeps running but is
    # recorded as a timeout; the root span is closed so the trace is still complete.
    def run_sync(self, task: dict) -> str:
        from langfuse import propagate_attributes

        with propagate_attributes(**self._attrs(task)):
            with self.lf.start_as_current_observation(name=self.meta.id, as_type="agent", input=task["input"]) as root:
                trace_id, t0 = self.lf.get_current_trace_id(), time.time()
                box: dict = {}

                import contextvars

                ctx = contextvars.copy_context()

                def call():
                    tools.set_run_id(self._run_id(task))
                    return self.mod.run(task)

                def target():
                    try:
                        box["out"] = ctx.run(call)
                    except BaseException as e:  # noqa: BLE001 - recorded on the trace
                        box["err"] = e

                th = threading.Thread(target=target, daemon=True)
                th.start()
                th.join(TASK_TIMEOUT_S)
                if th.is_alive():
                    return self._finish(task, root, trace_id, t0, error=TimeoutError(f"task exceeded {TASK_TIMEOUT_S}s"))
                if "err" in box:
                    return self._finish(task, root, trace_id, t0, error=box["err"])
                return self._finish(task, root, trace_id, t0, output=str(box.get("out")))

    async def run_async(self, task: dict) -> str:
        from langfuse import propagate_attributes

        with propagate_attributes(**self._attrs(task)):
            with self.lf.start_as_current_observation(name=self.meta.id, as_type="agent", input=task["input"]) as root:
                trace_id, t0 = self.lf.get_current_trace_id(), time.time()
                tools.set_run_id(self._run_id(task))  # this coroutine runs in its own Task context
                try:
                    out = await asyncio.wait_for(self.mod.arun(task), TASK_TIMEOUT_S)
                except BaseException as e:  # noqa: BLE001 - recorded on the trace
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    return self._finish(task, root, trace_id, t0, error=e)
                return self._finish(task, root, trace_id, t0, output=str(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="path to agents/<family>/<agent_id>")
    ap.add_argument("--batch", default=os.environ.get("PEIRA_BATCH", "dev"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("PEIRA_CONCURRENCY", "4")))
    ap.add_argument("--results-dir", default=os.environ.get("PEIRA_RESULTS_DIR", "results"))
    ap.add_argument("--rerun", action="store_true", help="ignore previous successes")
    args = ap.parse_args(argv)

    agent_dir = Path(args.agent).resolve()
    family = agent_dir.parent.name
    llm.export_provider_env()
    mod, meta = load_agent(agent_dir)
    tasks = load_tasks(agent_dir, args.limit)

    results_file = Path(args.results_dir) / args.batch / f"{meta.id}.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    skip = set() if args.rerun else done_ids(results_file)
    todo = [t for t in tasks if t["id"] not in skip]
    print(f"[{meta.id}] {len(todo)} to run, {len(skip)} already ok, concurrency={args.concurrency}", flush=True)

    if hasattr(mod, "setup"):
        mod.setup()
    runner = Runner(mod, meta, family, args.batch, results_file)

    statuses: list[str] = []
    try:
        if hasattr(mod, "arun"):
            async def go():
                sem = asyncio.Semaphore(args.concurrency)

                async def one(t):
                    async with sem:
                        return await runner.run_async(t)

                return await asyncio.gather(*(one(t) for t in todo))

            statuses = asyncio.run(go())
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                statuses = list(ex.map(runner.run_sync, todo))
    finally:
        tracing.shutdown()

    ok = statuses.count("ok")
    print(f"[{meta.id}] done: ok={ok} failed={len(statuses) - ok} (skipped {len(skip)})", flush=True)
    # Exit non-zero only if the whole batch failed: individual task failures are data, not crashes.
    return 0 if (ok > 0 or not todo) else 1


if __name__ == "__main__":
    sys.exit(main())
