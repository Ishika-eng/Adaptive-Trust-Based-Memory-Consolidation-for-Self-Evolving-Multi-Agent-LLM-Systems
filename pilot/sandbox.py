"""Runs untrusted, LLM-generated Python in an isolated subprocess.

Layers of isolation, outermost first:
  1. macOS `sandbox-exec` profile: network denied; file writes denied everywhere
     except the run's private temp directory. (Unavailable elsewhere -> the
     result reports isolation="process-limits-only".)
  2. A fresh temp working directory, deleted afterwards.
  3. Scrubbed environment: no API keys, HOME pointed at the temp directory.
  4. `python -I` (isolated mode), a wall-clock timeout that kills the whole
     process group, and CPU-time / file-size / open-file rlimits.
  5. stdout/stderr are truncated.

Not a hard security boundary against deliberately malicious code (reads are
allowed; macOS does not enforce RLIMIT_AS). For a hostile threat model, run
the whole thing in a container instead.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from config import ROOT

MAX_OUTPUT_CHARS = 4000
_SANDBOX_EXEC = shutil.which("sandbox-exec") if sys.platform == "darwin" else None


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    isolation: str


def _limits(cpu_s: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if sys.platform.startswith("linux"):
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    return apply


def _secret_paths() -> list[str]:
    home = Path.home()
    return [str(ROOT / ".env")] + [str(home / d) for d in (".ssh", ".aws", ".gnupg", ".config/gcloud")]


def _profile(workdir: str) -> str:
    secrets = "\n".join(f'(deny file-read* (subpath "{p}"))' for p in _secret_paths())
    return (
        "(version 1)\n(allow default)\n(deny network*)\n"
        f'(deny file-write* (require-not (require-any (subpath "{workdir}") '
        '(literal "/dev/null") (literal "/dev/dtracehelper"))))\n'
        f"{secrets}\n"
    )


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"


def run_python(code: str, timeout_s: float = 10.0) -> RunResult:
    # realpath: on macOS /var is a symlink to /private/var and the sandbox
    # profile matches real paths only.
    workdir = os.path.realpath(tempfile.mkdtemp(prefix="atmc_sbx_"))
    try:
        Path(workdir, "main.py").write_text(code)
        cmd = [sys.executable, "-I", "main.py"]
        isolation = "process-limits-only"
        if _SANDBOX_EXEC:
            profile = Path(workdir, "..", f"{Path(workdir).name}.sb").resolve()
            profile.write_text(_profile(workdir))
            cmd = [_SANDBOX_EXEC, "-f", str(profile)] + cmd
            isolation = "sandbox-exec"
        env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, preexec_fn=_limits(int(timeout_s) + 2),
        )
        try:
            out, err = proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            timed_out = True
        return RunResult(proc.returncode, _clip(out or ""), _clip(err or ""), timed_out, isolation)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            Path(workdir + ".sb").unlink()
        except FileNotFoundError:
            pass
