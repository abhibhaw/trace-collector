"""python_exec: run untrusted code in a resource-limited subprocess.

Layers: uid 65534 (cannot read the server's env or files), clean env, temp cwd, rlimits (AS, CPU, FSIZE, NPROC, NOFILE),
wall-clock kill of the whole process group, capped output, sockets disabled in-process, optional network namespace
(`unshare -rn`, when the kernel allows it), and the pod's egress NetworkPolicy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import shutil
import signal
import sys
import tempfile
import time

from registry import ToolError, tool

log = logging.getLogger("tools.sandbox")

MEM_BYTES = int(os.environ.get("SANDBOX_MEM_MB", "1024")) * 1024 * 1024
MAX_OUT = 10000
MAX_CODE = 100_000
NOBODY = 65534
SANDBOX_ROOT = os.environ.get("SANDBOX_ROOT", "/tmp/sbx")
_sem = asyncio.Semaphore(int(os.environ.get("SANDBOX_CONCURRENCY", "3")))
_unshare: list[str] | None = None

PRELUDE = r'''
import socket as _s, sys as _sys
def _blocked(*a, **k):
    raise OSError("network access is disabled in python_exec")
for _n in ("connect", "connect_ex", "sendto", "bind"):
    setattr(_s.socket, _n, _blocked)
_s.create_connection = _blocked
_s.getaddrinfo = _blocked
del _s, _n, _blocked
_sys.argv = ["main.py"]
_src = open("main.py", encoding="utf-8").read()
_g = {"__name__": "__main__", "__file__": "main.py", "__builtins__": __builtins__}
del _sys
exec(compile(_src, "main.py", "exec"), _g)
'''


def _limits(cpu_s: int, drop: bool) -> None:
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if drop:
        os.setgroups([])
        os.setgid(NOBODY)
        os.setuid(NOBODY)
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))


def _env(home: str) -> dict:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1",
            "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": home}


async def probe() -> None:
    """Decide once whether a network namespace is available for the sandbox."""
    global _unshare
    _unshare = []
    exe = shutil.which("unshare")
    if not exe:
        return
    drop = os.geteuid() == 0
    try:
        p = await asyncio.create_subprocess_exec(exe, "-rn", sys.executable, "-c", "pass", stdout=asyncio.subprocess.DEVNULL,
                                                 stderr=asyncio.subprocess.DEVNULL, preexec_fn=lambda: _limits(5, drop), env=_env("/tmp"))
        if await asyncio.wait_for(p.wait(), 10) == 0:
            _unshare = [exe, "-rn"]
    except Exception as e:  # noqa: BLE001
        log.info("unshare unavailable: %s", e)
    log.info("sandbox network namespace: %s", bool(_unshare))


async def _read_capped(stream: asyncio.StreamReader, cap: int) -> tuple[bytes, int]:
    buf, total = bytearray(), 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return bytes(buf), total
        total += len(chunk)
        if len(buf) < cap:
            buf.extend(chunk[: cap - len(buf)])


def _trunc(b: bytes, total: int) -> str:
    s = b.decode("utf-8", "replace")
    return s + (f"\n...[truncated, {total} bytes total]" if total > len(b) else "")


@tool("code")
async def python_exec(code: str, timeout_s: int = 20) -> dict:
    """Run Python 3.12 code in a sandbox (numpy, pandas, scipy, sympy available; no network)."""
    if not code.strip():
        raise ToolError("code must not be empty")
    if len(code) > MAX_CODE:
        raise ToolError(f"code is too long ({len(code)} chars, max {MAX_CODE})")
    timeout_s = max(1, min(int(timeout_s), 60))
    if _unshare is None:
        await probe()
    drop = os.geteuid() == 0
    async with _sem:
        os.makedirs(SANDBOX_ROOT, exist_ok=True)
        work = tempfile.mkdtemp(dir=SANDBOX_ROOT)
        try:
            with open(os.path.join(work, "main.py"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join(work, ".prelude.py"), "w", encoding="utf-8") as f:
                f.write(PRELUDE)
            if drop:
                os.chown(work, NOBODY, NOBODY)
            os.chmod(work, 0o700)
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *(_unshare or []), sys.executable, "-I", "-u", ".prelude.py", cwd=work, env=_env(work),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                preexec_fn=lambda: _limits(timeout_s, drop))
            timed_out = False
            readers = asyncio.gather(_read_capped(proc.stdout, MAX_OUT), _read_capped(proc.stderr, MAX_OUT))
            try:
                (out, out_n), (err, err_n) = await asyncio.wait_for(asyncio.shield(readers), timeout_s + 2)
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                (out, out_n), (err, err_n) = await asyncio.wait_for(readers, 5)
                await proc.wait()
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # stray children
            except (ProcessLookupError, PermissionError):
                pass
            rc = proc.returncode
            status = "timeout" if timed_out or rc == -signal.SIGXCPU else ("ok" if rc == 0 else "error")
            stderr = _trunc(err, err_n)
            if rc and rc < 0 and not stderr:
                stderr = f"process killed by signal {signal.Signals(-rc).name}" + (" (memory limit is 1 GB)" if -rc == signal.SIGKILL and not timed_out else "")
            return {"status": status, "exit_code": rc, "stdout": _trunc(out, out_n), "stderr": stderr,
                    "duration_s": round(time.monotonic() - t0, 3), "timeout_s": timeout_s}
        finally:
            shutil.rmtree(work, ignore_errors=True)
