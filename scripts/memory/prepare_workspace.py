"""Create a fresh experiment workspace with blank history or one frozen reference snapshot."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from Emerge.agent.reference_memory import ReferenceMemory, sha256
from Emerge.runtime.storage import atomic_json
from Emerge.runtime.workspace import reset_workspace_context
from Emerge.utils.helpers import sync_workspace_templates


def prepare(target: Path, source: Path | None = None, snapshot: str | None = None, *, blank=False, apply=False,
            read_only_memory=False):
    target = target.expanduser().resolve()
    if not blank and source is None:
        raise ValueError("Provide --source-workspace or choose --blank")
    source = source.expanduser().resolve() if source else None
    if blank and read_only_memory:
        raise ValueError('Read-only memory requires an existing reference snapshot')
    if target.exists():
        raise ValueError('Choose a new workspace path; existing workspaces are preserved')
    root = bundle = None
    if not blank:
        name = snapshot or json.loads((source / 'memory/reference/CURRENT.json').read_text())['snapshot']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Snapshot must be a name from memory/reference/snapshots')
        root = source / 'memory/reference/snapshots' / name
        bundle = json.loads((root / 'bundle.json').read_text())
        for relative in [*bundle['documents'], *(i['file'] for i in bundle['evidence'].values())]:
            if Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Snapshot files must use relative paths within the snapshot')
        for file, digest in bundle['documents'].items():
            ReferenceMemory._read(root, file, digest)
        for item in bundle['evidence'].values():
            ReferenceMemory._read(root, item['file'], item['sha256'])
    manifest = {'workspace': str(target), 'mode': 'blank' if blank else 'snapshot',
                'source_workspace': None if blank else str(source),
                'snapshot': root.name if root else None,
                'bundle_sha256': sha256((root / 'bundle.json').read_bytes()) if root else None,
                'documents': bundle['documents'] if bundle else {},
                'controller': bundle['controller'] if bundle else None,
                'history_policy': 'new sessions, plan, action queue and pending memory batches'}
    if read_only_memory:
        manifest['memory_policy'] = 'frozen_read_only'
    if not apply:
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.experiment-', dir=target.parent))
    try:
        sync_workspace_templates(staging, silent=True)
        # Runtime documents must start empty, rather than contain the bundled
        # example Go2/pet poses and scene objects. Controller publishes live state.
        reset_workspace_context(staging)
        if bundle:
            destination = staging / 'memory/reference'
            frozen = destination / 'snapshots' / manifest['snapshot']
            frozen.mkdir(parents=True)
            # Copy validated payloads, never old sessions/raw action history or symlinks.
            files = {'bundle.json', *bundle['documents'], *(i['file'] for i in bundle['evidence'].values())}
            for relative in files:
                output = frozen / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = (bundle['documents'].get(relative) or next(
                    (i['sha256'] for i in bundle['evidence'].values() if i['file'] == relative), None))
                raw = (root / relative).read_bytes()
                if digest and sha256(raw) != digest:
                    raise ValueError('Snapshot changed during copy')
                if relative == 'bundle.json' and sha256(raw) != manifest['bundle_sha256']:
                    raise ValueError('Snapshot bundle changed during copy')
                output.write_bytes(raw)
            for name in bundle['documents']:
                (destination / name).write_bytes((frozen / name).read_bytes())
            atomic_json(destination / 'CURRENT.json', {'snapshot': manifest['snapshot']})
            if read_only_memory:
                atomic_json(destination / 'REFERENCE_LOCK.json', {
                    'snapshot': manifest['snapshot'], 'bundle_sha256': manifest['bundle_sha256'],
                    'documents': manifest['documents'], 'memory_policy': 'frozen_read_only'})
        if read_only_memory:
            manifest['working_memory_files'] = {
                name: sha256((staging / name).read_bytes()) if (staging / name).is_file() else None
                for name in ('memory/MEMORY.md', 'memory/planning_state.json')}
        atomic_json(staging / 'EXPERIMENT_START.json', manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source-workspace', type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--blank', action='store_true')
    mode.add_argument('--snapshot', help='Snapshot name; defaults to source CURRENT')
    parser.add_argument('--apply', action='store_true', help='Create workspace; default validates/previews')
    parser.add_argument('--read-only-memory', action='store_true',
                        help='Pin the selected snapshot and disable all memory summarization writes')
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.source_workspace, args.snapshot,
                             blank=args.blank, apply=args.apply,
                             read_only_memory=args.read_only_memory), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
