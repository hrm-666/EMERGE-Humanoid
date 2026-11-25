"""Context builder for assembling agent prompts."""

import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from Emerge.agent.memory import MemoryStore
from Emerge.agent.reference_memory import ReferenceMemory
from Emerge.agent.skills import SkillsLoader
from Emerge.subagents.content import to_message_content
from Emerge.subagents.models import SubagentToolResult
from Emerge.utils.helpers import build_assistant_message


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    # Core orchestrator instructions — loaded when present.
    BOOTSTRAP_FILES = ["AGENTS.md"]

    # Embodied extension files — loaded only when present in workspace.
    # New tracks can add files here without touching other code.
    EMBODIED_FILES = [
        "EMBODIED.md",
        "ROBOT_STATE.md",
        "PLAN.md",
    ]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.reference_memory = ReferenceMemory(workspace)
        self.skills = SkillsLoader(workspace)
        self._active_skills: dict[Path, tuple[str, str]] = {}
        self._available_skills: dict[Path, str] = {}
        self._skills_summary = ""
        self._skills_initialized = False

    def prepare_skills(self, skill_names: list[str] | None = None) -> None:
        """Snapshot available instructions once per command, honoring workspace overrides."""
        self._active_skills.clear()
        self._available_skills = {
            Path(s['path']).resolve(): s['name'] for s in self.skills.list_skills()
        }
        self._skills_summary = self.skills.build_skills_summary()
        names = set(self.skills.get_always_skills()) | set(skill_names or ())
        for path, name in self._available_skills.items():
            if name in names:
                self.activate_skill(str(path))
        self._skills_initialized = True

    def _skill_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        return (candidate if candidate.is_absolute() else self.workspace / candidate).resolve()

    def activate_skill(self, path: str) -> bool:
        """Keep full, available skill contracts; failed reads never enter the registry."""
        resolved = self._skill_path(path)
        name = self._available_skills.get(resolved)
        if name and resolved not in self._active_skills:
            content = self.skills.load_skill(name)
            if content:
                self._active_skills[resolved] = (name, self.skills._strip_frontmatter(content))
                return True
        return False

    def active_skill_inventory(self) -> list[dict[str, str]]:
        return [{'name': name, 'path': str(path)} for path, (name, _) in self._active_skills.items()]

    def cached_skill_reply(self, path: str) -> str | None:
        entry = self._active_skills.get(self._skill_path(path))
        if entry:
            return (f'Skill {entry[0]} is already loaded in Active Skills for this command. '
                    'Use those instructions to choose the next action; no reread is needed.')
        return None

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills."""
        if not self._skills_initialized:
            self.prepare_skills(skill_names)
        parts = [self._get_identity()]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        if self._active_skills:
            content = '\n\n---\n\n'.join(
                f'### Skill: {name}\nSource: {path}\n\n{body}'
                for path, (name, body) in self._active_skills.items()
            )
            parts.append(f"# Active Skills\n\n{content}")

        skills_summary = self._skills_summary
        if skills_summary:
            parts.append(f"""# Skills

Active Skills above are already loaded in full and remain available throughout this command.
Use them directly to choose actions. Read SKILL.md only for an additional skill not yet loaded;
a successful read adds its full instructions to Active Skills. New commands refresh the skill files.
You can only use Skills with available="true".

{skills_summary}""")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        memory_entry = (
            "- Planning memory: memory/MEMORY.md (current task), memory/reference/SPACE.md and "
            "EXPERIENCES.md (canonical historical summaries). Use search_reference_memory tags "
            "for selected evidence. Full action/image archives are retrieved on demand."
        )

        platform_policy = ""
        if system == "Windows":
            platform_policy = """## Platform Policy (Windows)
- You are running on Windows. Do not assume GNU tools like `grep`, `sed`, or `awk` exist.
- Prefer Windows-native commands or file tools when they are more reliable.
- If terminal output is garbled, retry with UTF-8 output enabled.
"""
        else:
            platform_policy = """## Platform Policy (POSIX)
- You are running on a POSIX system. Prefer UTF-8 and standard shell tools.
- Use file tools when they are simpler or more reliable than shell commands.
"""

        return f"""# Emerge 🤖

You are Emerge, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
{memory_entry}
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

{platform_policy}

## Emerge Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Use the 'message' tool only for meaningful progress updates to the current user."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace.

        Core files (BOOTSTRAP_FILES) are always loaded.  Embodied extension
        files (EMBODIED_FILES) are loaded only when they exist, so new
        tracks can add protocol files without modifying this code.
        """
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        # Embodied extensions — present only when the controller is active
        for filename in self.EMBODIED_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call."""
        self.prepare_skills(skill_names)
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        merged = f"{runtime_ctx}\n\n{current_message}"
        content = merged
        if current_message != "[token-probe]" and self.reference_memory.installed:
            self.reference_memory.context(run_id=run_id, audit=True)

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": content},
        ]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str | SubagentToolResult,
    ) -> list[dict[str, Any]]:
        """Keep image blocks with their tool result; providers handle wire ordering."""
        if isinstance(result, SubagentToolResult):
            content = [{"type": "text", "text": result.text}, *to_message_content(result.content)]
        else:
            content = result
        if tool_name == 'read_file' and isinstance(result, str) and not result.startswith('Error'):
            call = next((c for m in reversed(messages) for c in m.get('tool_calls', [])
                         if c.get('id') == tool_call_id), None)
            if call:
                try:
                    path = json.loads(call['function']['arguments']).get('path', '')
                    if path.endswith('SKILL.md') and self.activate_skill(path):
                        # Also retain full contracts when compact planning context is disabled.
                        messages[0] = {'role': 'system', 'content': self.build_system_prompt()}
                except (KeyError, ValueError, TypeError):
                    pass
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": content,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list."""
        messages.append(
            build_assistant_message(
                content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )
        return messages
