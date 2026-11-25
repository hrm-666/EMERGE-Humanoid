"""Controller-scoped retrieval of persisted action text and original image bytes."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from Emerge.subagents import ImageContent, SubagentToolResult
from Emerge.utils.action_queue import parse_action_markdown


def controller_identity(workspace: Path) -> str:
    try:
        text = (workspace / "ROBOT_STATE.md").read_text()
        data = parse_action_markdown(text)
        if data is None:
            data = json.loads(text)
        robots = data.get("robots", {}) if isinstance(data, dict) else {}
        # New workspace templates contain illustrative, untyped robots. Once an
        # actual Controller publishes its identity, those examples are not owners.
        published = {key: robot for key, robot in robots.items()
                     if robot.get('control_owner') or robot.get('runtime') or robot.get('type')}
        robots = published or robots
        return "|".join(sorted(
            f"{key}:{robot.get('control_owner', robot.get('type', 'unknown'))}:{robot.get('runtime', 'unknown')}"
            for key, robot in robots.items())) or "workspace"
    except (OSError, ValueError, TypeError, AttributeError):
        return "workspace"


def audit_access(workspace: Path, event: str, **details) -> None:
    directory = workspace / "memory"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "access.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "at": datetime.now(timezone.utc).isoformat(),
                                 **details}, ensure_ascii=False) + "\n")


def _terms(text: str) -> set[str]:
    text = text.lower()
    # Shared task vocabulary lets Chinese requests retrieve English control actions.
    for words in (("苹果", "apple"), ("抽屉", "drawer"), ("柜子", "cabinet"),
                  ("抓取", "抓住", "grasp", "pick"), ("导航", "寻找", "navigate", "search"),
                  ("对位", "站位", "dock", "alignment"), ("释放", "放下", "release")):
        if any(word in text for word in words):
            text += " " + " ".join(words)
    tokens = set(re.findall(r"[a-z0-9]+", text))
    for part in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(part[i:i+2] for i in range(max(1, len(part)-1)))
    return tokens - {"the", "to", "and", "of", "in", "a", "is", "action"}


def retrieve(workspace: Path, query: str, *, limit: int = 5, scope: str = "controller",
             run_id: str | None = None, include_images: bool = True,
             task_run_id: str | None = None, sequence: bool = False,
             source: str = "tool") -> SubagentToolResult:
    """Keyword-ranked text retrieval; selected evidence images accompany the results."""
    identity = controller_identity(workspace)
    terms = _terms(query)
    if scope not in {"controller", "current_run", "previous_runs"}:
        raise ValueError("unknown memory scope")
    hits, eligible = [], []
    for path in (workspace / "memory/actions").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                continue
            if record.get("controller") != identity:
                continue
            if scope == "current_run" and (not run_id or record.get("run_id") != run_id):
                continue
            if scope == "previous_runs" and (not record.get("run_id") or record["run_id"] == run_id):
                continue
            if task_run_id is not None and record.get("run_id") != task_run_id:
                continue
            eligible.append(record)
            score = len(terms & _terms(record["text"]))
            if not score:
                continue
            hits.append((score, record.get("recorded_at", ""), record))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    hits.sort(key=lambda hit: (hit[0], hit[1]), reverse=True)
    limit = max(1, min(int(limit), 8))
    selected = [hit[2] for hit in hits[:limit]]
    sequence_note = ""
    if sequence and selected:
        anchor = selected[0]
        ordered = sorted((r for r in eligible if r.get("run_id") == anchor.get("run_id")),
                         key=lambda r: (r.get("started_at", r.get("recorded_at", "")), r["action_id"]))
        index = next(i for i, r in enumerate(ordered) if r["action_id"] == anchor["action_id"])
        start = max(0, min(index - limit // 2, len(ordered) - limit))
        selected = ordered[start:start + limit]
        previous = ordered[start - 1]["action_id"] if start else None
        end = start + len(selected)
        following = ordered[end]["action_id"] if end < len(ordered) else None
        sequence_note = (f"Chronological action window; run={anchor.get('run_id')}; "
                         f"anchor={anchor['action_id']}; previous={previous}; next={following}. "
                         "Query a boundary action ID with the same task_run_id to continue.\n")
    blocks, descriptions, paths = [], [], []
    for record in selected:
        descriptions.append(record["text"][:5000])
        # At most two images, with text labels binding pixels to action IDs.
        # Sequence views cover the route rather than spending both images
        # on the first two actions. Further before/after frames stay linked.
        frames = record.get("images", [])
        if sequence and len(selected) > 2:
            frames = frames[-1:]
        for relative in frames if include_images else []:
            if not isinstance(relative, str):
                continue
            if len(paths) >= 2:
                break
            path = (workspace / relative).resolve()
            if not path.is_relative_to((workspace / "memory/images").resolve()):
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            from Emerge.subagents.content import TextContent
            blocks.extend((TextContent(f"Historical action {record['action_id']}; image {relative}"),
                           ImageContent("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), detail="high")))
            paths.append(relative)
    audit_access(workspace, "retrieve", source=source, controller=identity, query=query,
                 scope=scope, run_id=run_id, task_run_id=task_run_id, sequence=sequence,
                 action_ids=[r["action_id"] for r in selected], images=paths)
    text = "Historical controller evidence (past scenes, not current state):\n" + sequence_note + (
        "\n\n".join(descriptions) if descriptions else "No matching archived action evidence.")
    return SubagentToolResult(text=text, content=tuple(blocks))
