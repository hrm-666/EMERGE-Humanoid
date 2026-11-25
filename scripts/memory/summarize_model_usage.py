"""Summarize per-attempt local usage, including visual calls and failed retries."""

import argparse
import json
from pathlib import Path


def summarize(workspace, run_ids=()):
    groups, seen = {}, set()
    path = workspace / 'memory/access.jsonl'
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row.get('event') != 'model_attempt' or run_ids and row.get('run_id') not in run_ids:
            continue
        key = (row['call_id'], row['attempt'])
        if key in seen:
            continue
        seen.add(key)
        group = groups.setdefault(row['scope'], {'calls': set(), 'attempts': 0, 'failed_attempts': 0,
            'retry_attempts': 0, 'usage_missing': 0, 'duration_ms': 0, 'usage': {}})
        group['calls'].add(row['call_id'])
        group['attempts'] += 1
        group['failed_attempts'] += row['status'] == 'failed'
        group['retry_attempts'] += row['attempt'] > 1
        group['usage_missing'] += row['usage_missing']
        group['duration_ms'] += row['duration_ms']
        for name, count in row['usage'].items():
            if isinstance(count, (int, float)):
                group['usage'][name] = group['usage'].get(name, 0) + count
    for group in groups.values():
        group['calls'] = len(group['calls'])
    return {'run_ids': list(run_ids), 'scopes': groups,
            'note': 'Known provider usage only; missing usage is not zero. Attempt time excludes retry backoff. Concurrent scopes may overlap.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-id', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(summarize(args.workspace, args.run_id), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
