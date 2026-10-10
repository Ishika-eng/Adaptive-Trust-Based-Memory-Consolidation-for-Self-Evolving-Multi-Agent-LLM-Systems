import os

import pytest

from config import ROOT
from sandbox import run_python

sandboxed = pytest.mark.skipif(run_python("print(1)").isolation != "sandbox-exec",
                               reason="needs macOS sandbox-exec")


def test_runs_plain_code():
    r = run_python("print(sum(range(10)))")
    assert r.returncode == 0 and r.stdout.strip() == "45"


def test_infinite_loop_is_killed():
    assert run_python("while True: pass", timeout_s=1.5).timed_out


def test_output_is_truncated():
    r = run_python("print('x' * 100000)")
    assert len(r.stdout) < 5000


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    r = run_python("import os; print(os.environ.get('GEMINI_API_KEY'))")
    assert "super-secret" not in r.stdout


@sandboxed
def test_cannot_write_outside_workdir(tmp_path):
    target = tmp_path / "pwned.txt"
    r = run_python(f"open({str(target)!r}, 'w').write('x')")
    assert r.returncode != 0 and not target.exists()


@sandboxed
def test_cannot_delete_files(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("keep")
    run_python(f"import os; os.remove({str(victim)!r})")
    assert victim.exists()


@sandboxed
def test_network_is_blocked():
    r = run_python("import socket; socket.create_connection(('1.1.1.1', 53), timeout=2); print('CONNECTED')")
    assert "CONNECTED" not in r.stdout


@sandboxed
def test_env_file_is_unreadable():
    env = ROOT / ".env"
    if not env.exists():
        pytest.skip("no .env present")
    r = run_python(f"print(open({str(env)!r}).read()[:10])")
    assert r.returncode != 0 and "GEMINI" not in r.stdout
