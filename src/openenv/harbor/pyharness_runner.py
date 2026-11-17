"""Fixed entry point of the `pyharness` seam, uploaded next to the harness. The harness cannot change it.

It loads `harness.py` from its own folder and calls `run(instruction, llm)`. `llm(messages, tools=None)` posts one
chat-completions request to the capture proxy and returns the assistant message as a dict. Once the model-call budget
is spent it raises `BudgetExhausted`, a `BaseException` so that a harness's `except Exception` cannot swallow it and
spin. Whatever else the harness raises is printed, which puts it in the rollout's `agent_log_tail`, and the rollout
still ends normally: the verifier grades whatever the harness left behind.

Standard library only, because it runs on the task image's own Python.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
import urllib.request
from pathlib import Path


class BudgetExhausted(BaseException):
    pass


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, max_turns: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_turns = max_turns
        self.calls = 0

    def __call__(self, messages: list, tools: list | None = None) -> dict:
        if self.max_turns and self.calls >= self.max_turns:
            raise BudgetExhausted(
                f"the budget of {self.max_turns} model calls is spent"
            )
        self.calls += 1
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)["choices"][0]["message"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-turns", type=int, default=0)
    args = parser.parse_args()

    # Popped, so the harness's tools (which inherit this environment) never see the endpoint or the session key
    llm = LLM(
        os.environ.pop("OPENAI_BASE_URL"),
        os.environ.pop("OPENAI_API_KEY"),
        args.model,
        args.max_turns,
    )
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))  # a harness may import its own helper files
    try:
        spec = importlib.util.spec_from_file_location("harness", here / "harness.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.run(Path(args.instruction_file).read_text(), llm)
    except BudgetExhausted as exc:
        print(f"[pyharness] stopped: {exc}")
    except Exception:
        print("[pyharness] the harness raised:")
        traceback.print_exc(file=sys.stdout)
    print(f"[pyharness] {llm.calls} model calls")


if __name__ == "__main__":
    main()
