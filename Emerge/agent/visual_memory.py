"""Bounded visual evidence buffers and a shared Markdown action journal."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps

from Emerge.agent.plan_state import CurrentSubgoal
from Emerge.runtime.storage import atomic_text
from Emerge.utils.helpers import safe_filename


@dataclass(frozen=True)
class EvidenceFrame:
    number: int
    revision: int
    episode: str
    captured_at: str
    images: dict[str, bytes]
    signatures: dict[str, np.ndarray]

    @classmethod
    def capture(cls, workspace: Path, manifest: dict, number: int, *, all_views: bool) -> EvidenceFrame:
        images, signatures = {}, {}
        for view in manifest["views"]:
            if not all_views and view["name"] != manifest["reference_view"]:
                continue
            # Read the actual bytes now; source files may be replaced or pruned during inference.
            data = (workspace / view["image_path"]).read_bytes()
            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                signatures[view["name"]] = np.asarray(image.resize((64, 64)), dtype=np.int16)
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=90)
                images[view["name"]] = buffer.getvalue()
        if not images:
            raise ValueError("Observation contains no selected camera views")
        return cls(
            number=number, revision=int(manifest["revision"]),
            episode=str(manifest.get("episode_id", "")),
            captured_at=str(manifest.get("captured_at", "unknown")),
            images=images, signatures=signatures,
        )

    def resembles(self, other: EvidenceFrame) -> bool:
        if self.images.keys() != other.images.keys():
            return False
        for view, signature in self.signatures.items():
            difference = np.abs(signature - other.signatures[view])
            # Require both a small average change and no strong local change.
            if difference.mean() >= 1.0 or difference.max() >= 12:
                return False
        return True


@dataclass
class ActionTrace:
    action: dict
    stage: CurrentSubgoal | None
    reasoning: str = ""
    frames: deque[EvidenceFrame] = field(default_factory=deque)
    first: EvidenceFrame | None = None
    latest: EvidenceFrame | None = None
    baseline: dict | None = None
    milestone: tuple[EvidenceFrame, dict] | None = None
    change: tuple[EvidenceFrame, dict] | None = None
    latest_review: tuple[EvidenceFrame, dict] | None = None
    reviewed: set[int] = field(default_factory=set)
    last_seen: tuple | None = None
    last_sample_revision: int = -1
    next_number: int = 0
    ended: bool = False
    interrupted: bool = False
    episode_changed: bool = False
    confirmations: int = 0
    note: str = ""

    def select_frames(self, limit: int = 4) -> list[EvidenceFrame]:
        candidates = {frame.number: frame for frame in self.frames if frame.number not in self.reviewed}
        for frame in (self.first, self.latest):
            if frame is not None and frame.number not in self.reviewed:
                candidates[frame.number] = frame
        ordered = sorted(candidates.values(), key=lambda frame: frame.number)
        if len(ordered) > limit:
            ordered = [ordered[round(index * (len(ordered) - 1) / (limit - 1))] for index in range(limit)]
        return ordered

    def evidence(self) -> list[tuple[EvidenceFrame, dict]]:
        """Keep start, a meaningful change, first success, and the final observed state."""
        selected = {}
        for frame, review, label in (
            (self.first, self.baseline, "Start of observation"),
            (*(self.change or (None, None)), "Key change"),
            (*(self.milestone or (None, None)), "Observed milestone"),
            (self.latest,
             self.latest_review[1] if self.latest_review and self.latest and self.latest_review[0].number == self.latest.number else None,
             "Last observed state"),
        ):
            if frame is None:
                continue
            if review is None:
                review = {"view": next(iter(frame.images)), "outcome": "uncertain",
                          "description": f"{label}; this frame has not been visually assessed."}
            selected[frame.number] = (frame, review)
        return [selected[number] for number in sorted(selected)]


class VisualMemory:
    """One monitor writes action entries; images become independent, permanent copies."""

    def __init__(self, workspace: Path):
        self.root = workspace / "memory"
        self.path = self.root / "MEMORY.md"

    def record(self, trace: ActionTrace) -> str:
        action_id = str(trace.action["id"])
        key = safe_filename(action_id)
        image_dir = self.root / "images" / key
        image_dir.mkdir(parents=True, exist_ok=True)
        evidence = trace.evidence()
        saved, descriptions = {}, []
        for frame, review in evidence:
            view = review["view"]
            data = frame.images[view]
            digest = hashlib.sha256(data).hexdigest()
            if digest not in saved:
                name = f"{frame.number:04d}_{safe_filename(view)}.jpg"
                (image_dir / name).write_bytes(data)
                saved[digest] = name
            relative = (Path("images") / key / saved[digest]).as_posix()
            description = " ".join(review["description"].split())
            description = description.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
            descriptions.append(
                f"- [{description}]({quote(relative, safe='/')}) — {review['outcome']}; "
                f"view `{view}`, revision {frame.revision}, captured {frame.captured_at}."
            )
        stage = trace.stage
        lines = [
            f"## Action {action_id}: {trace.action['action_type']}",
            f"- Execution status: {trace.action.get('status', 'unknown')}; "
            f"visual interrupt requested: {'yes' if trace.interrupted else 'no'}.",
            f"- Stage: {stage.step_id + ' — ' + stage.subgoal if stage else 'No active plan stage'}.",
            f"- Done criterion: {stage.done_criterion if stage else 'Not provided'}",
            f"- Plan revision: `{stage.plan_revision if stage else 'none'}`; "
            f"observation episode: `{trace.first.episode if trace.first else 'unknown'}`.",
            f"- Parameters: `{json.dumps(trace.action.get('parameters', {}), ensure_ascii=False)}`",
        ]
        if trace.reasoning:
            lines.append(f"- Action intent: {' '.join(trace.reasoning.split())}")
        if trace.action.get("result"):
            lines.append(f"- Controller result: {' '.join(str(trace.action['result']).split())[:2000]}")
        if trace.note:
            lines.append(f"- Observation note: {trace.note}")
        lines.extend(["", *descriptions] if descriptions else ["", "No camera evidence was captured for this action."])
        entry = "\n".join(lines) + "\n"
        start, end = f"<!-- visual-action:{key} -->", f"<!-- /visual-action:{key} -->"
        block = f"{start}\n{entry}{end}"
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else (
            "# Visual Action Memory\n\n"
            "Historical observations recorded by the visual monitor. Execution completion and "
            "visual milestones are not official task success. Image links are relative to this file.\n"
        )
        if start in existing:
            before, remainder = existing.split(start, 1)
            _, after = remainder.split(end, 1)
            updated = before + block + after
        else:
            updated = existing.rstrip() + "\n\n" + block + "\n"
        atomic_text(self.path, updated)
        # Only this action's generated images can be removed after the journal points at the new set.
        for image_path in image_dir.glob("*.jpg"):
            if image_path.name not in saved.values():
                image_path.unlink()
        return entry
