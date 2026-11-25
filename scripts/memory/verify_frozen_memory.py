"""Check pinned memory payloads and absence of summary writes before/after an F trial."""

import argparse
import json
from pathlib import Path

from Emerge.agent.reference_memory import ReferenceMemory, sha256


def verify(workspace: Path):
    workspace = workspace.expanduser().resolve()
    manifest = json.loads((workspace / 'EXPERIMENT_START.json').read_text())
    if manifest.get('memory_policy') != 'frozen_read_only':
        raise ValueError('Workspace is not a frozen-memory experiment')
    reference = workspace / 'memory/reference'
    lock = json.loads((reference / 'REFERENCE_LOCK.json').read_text())
    for key in ('snapshot', 'bundle_sha256', 'documents', 'memory_policy'):
        if lock.get(key) != manifest[key]:
            raise ValueError(f'Frozen lock differs from experiment manifest: {key}')
    if json.loads((reference / 'CURRENT.json').read_text())['snapshot'] != manifest['snapshot']:
        raise ValueError('CURRENT changed during frozen experiment')
    root = reference / 'snapshots' / manifest['snapshot']
    if sha256((root / 'bundle.json').read_bytes()) != manifest['bundle_sha256']:
        raise ValueError('Frozen bundle changed')
    bundle = json.loads((root / 'bundle.json').read_text())
    for name, digest in manifest['documents'].items():
        ReferenceMemory._read(root, name, digest)
        if sha256((reference / name).read_bytes()) != digest:
            raise ValueError(f'Projected memory document changed: {name}')
    for item in bundle['evidence'].values():
        ReferenceMemory._read(root, item['file'], item['sha256'])
    if {p.name for p in (reference / 'snapshots').iterdir()} != {manifest['snapshot']}:
        raise ValueError('Unexpected reference snapshot added')
    for name, digest in manifest['working_memory_files'].items():
        path = workspace / name
        actual = sha256(path.read_bytes()) if path.is_file() else None
        if actual != digest:
            raise ValueError(f'Working memory was written: {name}')
    runs = 0
    for path in (workspace / 'runs').glob('*/events.jsonl'):
        runs += 1
        for event in map(json.loads, path.read_text().splitlines()):
            if event['type'] in ('memory.updated', 'memory.update_usage'):
                raise ValueError(f'Memory update occurred in {path.parent.name}')
    access = workspace / 'memory/access.jsonl'
    if access.is_file():
        for event in map(json.loads, access.read_text().splitlines()):
            if event.get('event') == 'model_attempt' and event.get('scope') == 'planning_memory':
                raise ValueError('Planning-memory API was called')
    return {'status': 'verified', 'workspace': str(workspace), 'snapshot': manifest['snapshot'],
            'bundle_sha256': manifest['bundle_sha256'], 'evidence_count': len(bundle['evidence']),
            'runs': runs, 'summary_writes': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.workspace), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
