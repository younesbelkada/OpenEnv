"""The `pyharness` seam: a harness sent per rollout through `harness_config`, run by a fixed runner."""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("harbor")

from openenv.harbor.pyharness import PyHarnessAgent, RUNNER  # noqa: E402
from openenv.harbor.rollout import build_trial_config  # noqa: E402

HARNESS = (
    "def run(instruction, llm):\n    llm([{'role': 'user', 'content': instruction}])\n"
)


def test_harness_config_reaches_the_agent_kwargs(tmp_path):
    config = build_trial_config(
        task_dir=tmp_path,
        harness="pyharness",
        sandbox="docker",
        intercept_url="https://capture.example",
        session_id="session-1",
        model="LiquidAI/LFM3-1.7B",
        trial_name="trial",
        trials_dir=tmp_path,
        agent_step_limit=17,
        harness_config={"harness_files": {"harness.py": HARNESS}},
    )
    assert config.agent.import_path == "openenv.harbor.pyharness:PyHarnessAgent"
    assert config.agent.kwargs == {
        "max_turns": 17,
        "harness_files": {"harness.py": HARNESS},
    }
    assert config.agent.env == {
        "OPENAI_API_KEY": "session-1",
        "OPENAI_BASE_URL": "https://capture.example/v1",
    }


def test_harness_files_must_stay_in_the_harness_folder(tmp_path):
    with pytest.raises(ValueError, match="harness.py"):
        PyHarnessAgent(logs_dir=tmp_path, harness_files={"other.py": ""})
    for path in ("../escape.py", "/etc/passwd", "runner.py"):
        with pytest.raises(ValueError, match="relative path"):
            PyHarnessAgent(
                logs_dir=tmp_path, harness_files={"harness.py": HARNESS, path: ""}
            )


class _Engine(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Engine.requests.append((self.headers["Authorization"], body))
        reply = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.mark.parametrize(
    "harness, expected",
    [
        (HARNESS, "1 model calls"),
        # `except Exception` must not swallow the budget, or the harness spins until the timeout
        (
            "def run(instruction, llm):\n    while True:\n        try:\n            llm([])\n"
            "        except Exception:\n            pass\n",
            "stopped: the budget of 2 model calls is spent",
        ),
        (
            "def run(instruction, llm):\n    raise KeyError('boom')\n",
            "KeyError: 'boom'",
        ),
    ],
)
def test_runner(tmp_path, harness, expected):
    server = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Engine.requests = []
    (tmp_path / "runner.py").write_text(RUNNER.read_text())
    (tmp_path / "harness.py").write_text(harness)
    (tmp_path / "instruction.md").write_text("Write 42 to /workdir/answer.txt")
    env = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "OPENAI_API_KEY": "session-1",
    }
    out = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "runner.py"),
            "--instruction-file",
            str(tmp_path / "instruction.md"),
            "--model",
            "m",
            "--max-turns",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    ).stdout
    server.shutdown()
    assert expected in out
    assert all(auth == "Bearer session-1" for auth, _ in _Engine.requests)
    assert all(body["model"] == "m" for _, body in _Engine.requests)
    assert len(_Engine.requests) <= 2
