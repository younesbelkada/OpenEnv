"""Harbor agent for the `pyharness` seam: a Python agent loop that arrives with each rollout.

The harness is data, not an install. `harness_files` maps relative paths to source text, must include `harness.py`,
and reaches the agent per rollout through `run_rollout(harness_config={"harness_files": ...})`. The files are
uploaded next to the fixed runner (`pyharness_runner.py`) before the agent runs, so a trainer can change the harness
between rollouts with no image rebuild and no server restart.

The runner owns what a harness must not change: the model endpoint, the model-call budget and the entry point.
`harness.py` defines `run(instruction, llm)` and owns everything else: prompt, tools, tool execution, context and
stopping. The runner's output goes to /logs/agent/agent.log, whose tail comes back as `agent_log_tail`.
"""

from __future__ import annotations

import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

ROOT = "/opt/pyharness"
RUNNER = Path(__file__).with_name("pyharness_runner.py")


class PyHarnessAgent(BaseInstalledAgent):
    def __init__(
        self,
        *args: Any,
        harness_files: dict[str, str] | None = None,
        max_turns: int = 0,
        **kwargs: Any,
    ) -> None:
        if not harness_files or "harness.py" not in harness_files:
            raise ValueError(
                "pyharness needs harness_config={'harness_files': {...}} with a harness.py"
            )
        for path in harness_files:
            parts = PurePosixPath(path).parts
            if (
                PurePosixPath(path).is_absolute()
                or ".." in parts
                or path == "runner.py"
            ):
                raise ValueError(
                    f"harness file {path!r} must be a relative path inside the harness folder"
                )
        self._harness_files = harness_files
        self._max_turns = max_turns
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "pyharness"

    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(environment, command=f"mkdir -p {ROOT}")
        with tempfile.TemporaryDirectory(prefix="pyharness-") as tmp:
            files = {"runner.py": RUNNER.read_text(), **self._harness_files}
            for path, text in files.items():
                local = Path(tmp, path)
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(text)
                await self.exec_as_root(
                    environment,
                    command=f"mkdir -p {shlex.quote(str(PurePosixPath(ROOT, path).parent))}",
                )
                await environment.upload_file(local, str(PurePosixPath(ROOT, path)))
        await self.exec_as_root(environment, command=f"chmod -R a+rX {ROOT}")

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        local = self.logs_dir / "instruction.md"
        local.write_text(instruction)
        await environment.upload_file(local, f"{ROOT}/instruction.md")
        await self.exec_as_root(environment, command=f"chmod a+r {ROOT}/instruction.md")
        model = (self.model_name or "").split("/", 1)[-1]
        # Into /logs/agent, which Harbor copies back even after a timeout: the rollout's `agent_log_tail`
        await self.exec_as_agent(
            environment,
            command=(
                f"python3 -u {ROOT}/runner.py --instruction-file {ROOT}/instruction.md "
                f"--model {shlex.quote(model)} --max-turns {self._max_turns} > /logs/agent/agent.log 2>&1"
            ),
        )
