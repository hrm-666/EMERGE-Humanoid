"""Build a synthetic legacy review package for offline compatibility tests."""
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["observed", "inferred", "uncertain"]
ShortText = Annotated[str, Field(min_length=1, max_length=400)]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence: Confidence
    evidence_ids: list[str] = Field(
        min_length=1, max_length=8,
        description="优先1–2个关键证据；跨任务比较可用2–4个，尽量覆盖两项任务。",
    )


class SpatialFact(Row):
    kind: Literal["object", "relation"]
    entities: list[str] = Field(min_length=1, max_length=2)
    statement: ShortText


class Experience(Row):
    stage: str
    conditions: ShortText
    effective: ShortText
    ineffective: ShortText
    recommendation: ShortText


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    space: list[SpatialFact] = Field(max_length=30)
    experiences: list[Experience] = Field(max_length=8)


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".review-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_review(data, allowed_ids):
    review = Review.model_validate(data)
    seen = set()
    for item in review.space:
        expected = 1 if item.kind == "object" else 2
        if len(set(item.entities)) != expected or len(item.entities) != expected:
            raise ValueError("Spatial key must have one object or two distinct objects")
        if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", entity) for entity in item.entities):
            raise ValueError("Use stable lowercase entity IDs")
        item.entities.sort()
        key = (item.kind, *item.entities)
        if key in seen:
            raise ValueError(f"Duplicate spatial relationship: {key}")
        seen.add(key)
    if len({item.stage for item in review.experiences}) != len(review.experiences):
        raise ValueError("Duplicate experience stage")
    for item in [*review.space, *review.experiences]:
        unknown = set(item.evidence_ids) - set(allowed_ids)
        if unknown:
            raise ValueError(f"Unknown or unseen evidence IDs: {sorted(unknown)}")
        item.evidence_ids = sorted(set(item.evidence_ids))
    return review.model_dump()


def render(review, evidence, snapshot):
    source_labels = {eid: f"E{index:03d}" for index, eid in enumerate(sorted(evidence), 1)}

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    def sources(ids):
        return ", ".join(source_labels[eid] for eid in ids)

    header = (f"\n快照：`{snapshot}`。仅含所选历史记录；尚未接入 EMERGE。\n"
              "\n短编号的图片、记录和时间见 [证据索引](EVIDENCE.md)，需要复核时再查阅。\n")
    space = "# 空间记忆\n" + header + "\n物体位置为最近历史观察，重置后需重新确认。\n\n"
    space += "| 唯一键 | 当前综合结论 | 证据性质 | 证据编号 |\n| --- | --- | --- | --- |\n"
    for item in sorted(review["space"], key=lambda row: (row["kind"], row["entities"])):
        values = [f"{item['kind']}:{'/'.join(item['entities'])}", item["statement"],
                  item["confidence"], sources(item["evidence_ids"])]
        space += "| " + " | ".join(map(cell, values)) + " |\n"
    experience = "# 执行经验\n" + header + "\n按阶段综合所选记录；建议不等于已经验证。\n\n"
    for item in review["experiences"]:
        experience += f"## {item['stage']}\n\n"
        for key, label in (("conditions", "条件"), ("effective", "有效分支"),
                           ("ineffective", "无效尝试"), ("recommendation", "下次选择")):
            experience += f"- {label}：{item[key]}\n"
        experience += f"- 证据：{item['confidence']}；{sources(item['evidence_ids'])}\n\n"
    ids = sorted({eid for row in review["space"] + review["experiences"] for eid in row["evidence_ids"]})
    index = (f"# 证据索引\n\n快照：`{snapshot}`。按需查阅，不作为默认记忆正文。\n\n"
             "图片时间采用动作记录时间；事件编号 seq 指向 JSONL 中对应事件。\n"
             "完整元数据及来源哈希保存在对应快照的 manifest.json。\n\n")
    for eid in ids:
        entry = evidence[eid]
        index += (f"## {source_labels[eid]}\n\n"
                  f"- ID：`{eid}`\n- 时间：{entry['time']}\n"
                  f"- 来源：[打开记录或图片]({entry['path']})\n\n")
    return {"SPACE.md": space, "EXPERIENCES.md": experience, "EVIDENCE.md": index}


from Emerge.agent.action_memory import audit_access, controller_identity
from Emerge.agent.reference_memory import sha256
import shutil


def build_package(source: Path):
    source_snapshot = json.loads((source / "CURRENT.json").read_text())["snapshot"]
    snapshot = source / "snapshots" / source_snapshot
    manifest = json.loads((snapshot / "manifest.json").read_text())
    evidence = manifest["evidence"]
    review = validate_review(json.loads((snapshot / "memory.json").read_text()), evidence)
    labels = {eid: f"E{index:03d}" for index, eid in enumerate(sorted(evidence), 1)}
    selected = {eid for row in review["space"] + review["experiences"] for eid in row["evidence_ids"]}
    payloads, installed, controllers = {}, {}, set()
    event_cache = {}
    queue = sorted(selected)
    while queue:
        eid = queue.pop(0)
        label = labels[eid]
        if label in installed:
            continue
        item = evidence[eid]
        kind, run, action = eid.split(":", 2)
        path = Path(item["path"])
        raw = path.read_bytes()
        expected = manifest["source_sha256"].get(str(path))
        if expected and sha256(raw) != expected:
            raise ValueError(f"Source changed: {path}")
        image_labels = []
        if kind == "action":
            record = json.loads(raw)
            if record["run_id"] != run or record["action_id"] != action:
                raise ValueError(f"Action identity mismatch: {eid}")
            controllers.add(record["controller"])
            companions = [key for key in evidence if key.startswith(f"image:{run}:{action}:")]
            image_labels = [labels[key] for key in companions]
            queue.extend(companions)
        elif kind == "image":
            action_eid = f"action:{run}:{action.split(':')[0]}"
            if action_eid not in evidence:
                raise ValueError(f"Missing action for image {eid}")
            queue.append(action_eid)
        elif kind == "event":
            if path not in event_cache:
                event_cache[path] = [json.loads(line) for line in raw.decode().splitlines() if line]
            event = next(e for e in event_cache[path] if str(e["seq"]) == action)
            call_id = event.get("data", {}).get("tool_call_id")
            start = next((e for e in event_cache[path]
                          if e["type"] == "tool.started" and e.get("data", {}).get("tool_call_id") == call_id), None)
            raw = encode({"call": start, "result": event}).encode()
        else:
            raise ValueError(f"Unsupported evidence kind: {kind}")
        relative = f"evidence/{label}.{'jpg' if kind == 'image' else 'json'}"
        payloads[relative] = raw
        installed[label] = {"evidence_id": eid, "kind": kind, "run_id": run,
                            "time": item["time"], "file": relative, "sha256": sha256(raw),
                            "image_labels": image_labels}
    if len(controllers) != 1:
        raise ValueError(f"Expected one controller, got {controllers}")
    rows = []
    for row in review["space"]:
        rows.append({**row, "id": row["kind"] + ":" + "/".join(row["entities"]),
                     "tags": row["entities"], "evidence_ids": [labels[eid] for eid in row["evidence_ids"]]})
    for row in review["experiences"]:
        rows.append({**row, "id": "stage:" + row["stage"], "tags": row["stage"].split("_"),
                     "evidence_ids": [labels[eid] for eid in row["evidence_ids"]]})
    # An explicit image tag also resolves its action text via the same installed archive.
    for item in installed.values():
        if item["kind"] == "image":
            rid, aid = item["evidence_id"].split(":")[1:3]
            item["action_label"] = labels[f"action:{rid}:{aid}"]
    docs = render(review, evidence, source_snapshot)
    for name, text in docs.items():
        if name == "EVIDENCE.md":
            for eid in selected:
                text = text.replace(evidence[eid]["path"], installed[labels[eid]]["file"])
        text = text.replace("尚未接入 EMERGE", "EMERGE 只读历史参考")
        payloads[name] = text.encode()
    bundle = {"schema_version": 1, "snapshot": source_snapshot, "controller": controllers.pop(),
              "rows": rows, "evidence": installed,
              "documents": {name: sha256(payloads[name]) for name in ("SPACE.md", "EXPERIENCES.md")}}
    payloads["bundle.json"] = encode(bundle).encode()
    identity = sha256(encode({name: sha256(raw) for name, raw in sorted(payloads.items())}).encode())[:20]
    return identity, payloads, bundle


def install(source: Path, workspace: Path, apply: bool):
    # The reusable package survives an independent-trial workspace reset.
    package_index = source / "reference_bundle/CURRENT.json"
    try:
        identity, payloads, bundle = build_package(source)
    except FileNotFoundError:
        if not package_index.exists():
            raise
        identity = json.loads(package_index.read_text())["snapshot"]
        package = source / "reference_bundle/snapshots" / identity
        payloads = {str(path.relative_to(package)): path.read_bytes()
                    for path in package.rglob("*") if path.is_file()}
        bundle = json.loads(payloads["bundle.json"])
        requested = json.loads((source / "CURRENT.json").read_text())["snapshot"]
        if bundle["snapshot"] != requested:
            raise ValueError("Original evidence is missing and the reusable package is for an older summary")
        print("Using the saved self-contained package; original evidence files are unavailable.")
    active = controller_identity(workspace)
    if active not in ("workspace", bundle["controller"]):
        raise ValueError(f"Reference controller {bundle['controller']} differs from {active}")
    print(encode({"mode": "apply" if apply else "preview", "snapshot": identity,
                  "source_snapshot": bundle["snapshot"], "controller": bundle["controller"],
                  "summary_rows": len(bundle["rows"]), "evidence_files": len(bundle["evidence"]),
                  "bytes": sum(len(raw) for raw in payloads.values()),
                  "destination": str(workspace / "memory/reference")}))
    if not apply:
        return
    package = source / "reference_bundle/snapshots" / identity
    for name, raw in payloads.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    atomic_write(package_index, encode({"snapshot": identity}))
    target = workspace / "memory/reference/snapshots" / identity
    shutil.copytree(package, target, dirs_exist_ok=True)
    atomic_write(workspace / "memory/reference/CURRENT.json", encode({"snapshot": identity}))
    for name in ("SPACE.md", "EXPERIENCES.md", "EVIDENCE.md"):
        # Root reading copies need links relative to the installed snapshot.
        text = payloads[name].decode()
        if name == "EVIDENCE.md":
            text = text.replace("](evidence/", f"](snapshots/{identity}/evidence/")
        atomic_write(workspace / "memory/reference" / name, text)
    audit_access(workspace, "reference_installed", snapshot=bundle["snapshot"],
                 package=identity, controller=bundle["controller"],
                 documents=bundle["documents"], reference_ids=[row["id"] for row in bundle["rows"]])
    print("Read-only reference installed. Start a new EMERGE session; no API calls performed.")
