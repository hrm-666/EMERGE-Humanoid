"""Read-only summarized memory, with explicit label-based evidence expansion."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from Emerge.agent.action_memory import _terms, audit_access, controller_identity
from Emerge.subagents import ImageContent, SubagentToolResult
from Emerge.subagents.content import TextContent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cited_labels(text, evidence):
    """Resolve existing inline citations without inventing or renumbering labels."""
    return list(dict.fromkeys(label for label in re.findall(r'(?<![A-Za-z0-9_])E\d+(?![A-Za-z0-9_])', text or '')
                              if label in evidence))


def row_evidence_labels(row, evidence, query=""):
    """Put the successful route's cited views ahead of legacy row references."""
    segments = []
    for sentence in re.split(r'[；;。\n]', row.get('effective', '')):
        context = []
        for part in re.split(r'→|->', sentence):
            context.append(part)
            if cited_labels(part, evidence):
                segments.append('→'.join(context))
                context = []
    if query:
        terms = _terms(query)
        segments = sorted(segments, key=lambda text: -len(terms & _terms(text)))
    route = [label for segment in segments for label in cited_labels(segment, evidence)]
    return list(dict.fromkeys(route + row.get('evidence_ids', [])))


def grouped_evidence(labels, evidence):
    """Keep each ranked citation beside its action and entry/arrival frames."""
    ordered = []
    for label in labels:
        item = evidence[label]
        action = item.get('action_label')
        frames = item.get('image_labels', [])
        frames = list(dict.fromkeys(frames[:1] + frames[-1:]))
        # For an explicit image, that image remains first in its group.
        companions = ([action] if action else []) + frames
        if action in evidence:
            related = evidence[action].get('image_labels', [])
            companions += list(dict.fromkeys(related[:1] + related[-1:]))
        for key in [label, *companions]:
            if key in evidence and key not in ordered:
                ordered.append(key)
    return ordered


class ReferenceMemory:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.directory = workspace / "memory/reference"

    @property
    def installed(self):
        return (self.directory / "CURRENT.json").is_file()

    @property
    def read_only(self):
        return (self.directory / "REFERENCE_LOCK.json").is_file()

    def load(self):
        lock = self.directory / "REFERENCE_LOCK.json"
        current = json.loads((lock if lock.is_file() else self.directory / "CURRENT.json").read_text())
        name = current["snapshot"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError("Invalid reference snapshot")
        root = self.directory / "snapshots" / name
        raw = (root / "bundle.json").read_bytes()
        if lock.is_file() and sha256(raw) != current['bundle_sha256']:
            raise ValueError("Frozen reference bundle changed")
        bundle = json.loads(raw)
        identity = controller_identity(self.workspace)
        if identity != bundle["controller"]:
            raise ValueError(f"Reference controller mismatch: {identity}")
        return root, bundle

    @staticmethod
    def _read(root, relative, expected_sha):
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Evidence path outside installed snapshot")
        data = path.read_bytes()
        if sha256(data) != expected_sha:
            raise ValueError(f"Reference file changed: {relative}")
        return data

    def context(self, *, run_id=None, audit=False):
        try:
            root, bundle = self.load()
            parts = []
            for name in ("SPACE.md", "EXPERIENCES.md"):
                parts.append(f"### memory/reference/{name}\n\n" + self._read(
                    root, name, bundle["documents"][name]).decode())
            if audit:
                audit_access(self.workspace, "reference_context", run_id=run_id,
                             snapshot=bundle["snapshot"], controller=bundle["controller"],
                             documents=bundle["documents"],
                             reference_ids=[row["id"] for row in bundle["rows"]], images=[])
            return ("## Summarized spatial and execution memory\n"
                    "Use these historical summaries to choose routes and actions, together with current observations. "
                    "Tags such as stage:table_to_cabinet, object:apple and E423 resolve through "
                    "search_reference_memory. Start with summaries; expand evidence only when useful. "
                    "When a memory informs a decision, mention its tag and the resulting choice in "
                    "update_plan.change_summary or execute_robot_action.reasoning. "
                    "Object positions may have changed or reset; active Skills govern current action contracts.\n\n"
                    + "\n\n".join(parts))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if audit:
                audit_access(self.workspace, "reference_error", run_id=run_id, error=str(exc))
            return f"Reference memory unavailable: {exc}. Use current observations."

    def search(self, query="", *, reference_ids=None, include_evidence=False,
               include_images=True, limit=4, run_id=None, reuse=None, reread_images=False):
        try:
            root, bundle = self.load()
            rows = {row["id"]: row for row in bundle["rows"]}
            references = list(dict.fromkeys(reference_ids or []))
            if include_evidence and not references:
                return SubagentToolResult(text="Choose a returned memory tag or E number in reference_ids to expand evidence.")
            selected, labels = [], []
            if references:
                for reference in references:
                    key = reference if reference in rows else f"stage:{reference}"
                    if key in rows:
                        selected.append(rows[key])
                        labels.extend(row_evidence_labels(rows[key], bundle["evidence"], query))
                    elif reference.upper() in bundle["evidence"]:
                        labels.append(reference.upper())
                    else:
                        return SubagentToolResult(text=f"Error: unknown memory reference {reference!r}; search summaries first.")
            else:
                terms = _terms(query)
                scored = [(len(terms & _terms(encode_row(row))), row) for row in rows.values()]
                selected = [row for score, row in sorted(scored, key=lambda item: (-item[0], item[1]["id"]))
                            if score][:max(1, min(limit, 8))]
                labels = [label for row in selected
                          for label in row_evidence_labels(row, bundle["evidence"], query)]
            selected = list({row["id"]: row for row in selected}.values())
            # An explicitly requested frame/action is more specific than a whole stage.
            explicit = [ref.upper() for ref in references if ref.upper() in bundle["evidence"]]
            labels = list(dict.fromkeys(explicit + labels))
            text = ["Historical summarized memory; apply with current observations.",
                    *[encode_row(row) for row in selected]]
            if not selected and not labels:
                text.append("No matching summarized memory. Explore from the current scene.")
            if labels:
                text.append("Evidence tags: " + ", ".join(labels))
            summary_end = len(text)
            blocks, opened, images, errors = [], [], [], []
            reused_images, reused_records, candidate_images = [], [], []
            if include_evidence:
                expanded = grouped_evidence(labels, bundle['evidence'])
                counts = {'image': 0, 'record': 0}
                package = []
                for label in expanded:
                    kind = 'image' if bundle['evidence'][label]['kind'] == 'image' else 'record'
                    cap = (2 if include_images else 0) if kind == 'image' else 4
                    if counts[kind] < cap:
                        package.append(label)
                        counts[kind] += 1
                seen_images = reuse.setdefault('images', {}) if reuse is not None else {}
                seen_records = reuse.setdefault('records', set()) if reuse is not None else set()
                sent_images = set()
                for label in package:
                    item = bundle["evidence"][label]
                    digest = item['sha256']
                    if item['kind'] == 'image':
                        candidate_images.append(label)
                        if digest in sent_images or (digest in seen_images and not reread_images):
                            reused_images.append(label)
                            continue
                    elif digest in seen_records:
                        reused_records.append(label)
                        continue
                    try:
                        raw = self._read(root, item["file"], item["sha256"])
                        if item["kind"] == "image":
                            blocks.extend((TextContent(f"Historical evidence {label}: {item['evidence_id']}"),
                                           ImageContent("data:image/jpeg;base64," + base64.b64encode(raw).decode(),
                                                        detail="high")))
                            images.append(label)
                            seen_images[digest] = label
                            sent_images.add(digest)
                        else:
                            record = json.loads(raw)
                            if item["kind"] == "action":
                                detail = "\n".join(line for line in record["text"].splitlines()
                                                   if not re.search(r"\]\((?:images|memory/images)/", line))
                            else:
                                detail = json.dumps(record, ensure_ascii=False)
                            text.append(f"[{label}] {item['evidence_id']}\n{detail[:5000]}")
                            seen_records.add(digest)
                        opened.append(label)
                    except (OSError, ValueError, KeyError) as exc:
                        errors.append(f"{label}: {exc}")
                if reused_images or reused_records:
                    # The canonical summaries are already in the planning context.
                    # On reuse, return only new records and the retained decision.
                    text = [text[0], *text[summary_end:]]
                    text.append('Reuse already selected evidence: ' + ', '.join(reused_images + reused_records))
                    comparison = (reuse or {}).get('comparison')
                    if comparison:
                        text.append('Retained comparison: ' + json.dumps(comparison, ensure_ascii=False))
                    if (reuse or {}).get('route'):
                        text.append('Pending step: ' + reuse['route'])
                    if (reuse or {}).get('segment_event') == 'route_invalidated':
                        text.append('Previous route invalidated; select another relevant exit or evidence tag from the summaries.')
                    elif not comparison:
                        text.append('Use the retained route and current view; save the comparison with the next action.')
            text.extend(errors)
            reused = include_evidence and not opened and not errors and (reused_images or reused_records)
            audit_access(self.workspace, 'reference_reused' if reused else
                         ("reference_evidence" if include_evidence else "reference_search"),
                         run_id=run_id, snapshot=bundle["snapshot"], query=query,
                         requested_reference_ids=references, reference_ids=[row["id"] for row in selected],
                         evidence_ids=opened, images=images, errors=errors,
                         candidate_images=candidate_images, reused_images=reused_images,
                         reused_records=reused_records, reread_images=reread_images,
                         segment=(reuse or {}).get('segment'),
                         comparison=(reuse or {}).get('comparison'))
            return SubagentToolResult(text="\n\n".join(text), content=tuple(blocks))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audit_access(self.workspace, "reference_error", run_id=run_id, error=str(exc))
            return SubagentToolResult(text=f"Error: summarized memory unavailable: {exc}")

    def record_decision(self, tool, arguments, run_id=None):
        """Record model-declared use; it is not a certificate of a correct decision."""
        if not self.installed or tool not in ("update_plan", "execute_robot_action"):
            return None
        try:
            _, bundle = self.load()
            reason = arguments.get("reasoning", "") if tool == "execute_robot_action" else arguments.get("change_summary", "")
            if tool == 'execute_robot_action' and arguments.get('route_progress'):
                reason += ' ' + arguments['route_progress'].get('next_route', '')
                reason += ' ' + json.dumps(arguments['route_progress'].get('comparison', {}), ensure_ascii=False)
            if tool == 'update_plan' and arguments.get('exploration'):
                reason += ' ' + json.dumps(arguments['exploration'].get('comparison', {}), ensure_ascii=False)
            candidates = [row["id"] for row in bundle["rows"]] + list(bundle["evidence"])
            used = [tag for tag in candidates if re.search(r"(?<![A-Za-z0-9_:/])" + re.escape(tag) + r"(?![A-Za-z0-9_:/])", reason)]
            if used:
                data = {"run_id": run_id, "snapshot": bundle["snapshot"], "tool": tool,
                        "reference_ids": used, "reasoning": reason, "basis": "model_declared"}
                audit_access(self.workspace, "reference_decision", **data)
                return data
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None


def encode_row(row):
    return json.dumps(row, ensure_ascii=False)
