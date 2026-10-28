"""Make a runs root for the explanation pipeline whose bindings differ from the originals in paths only.

For every run under SRC (top<K>/<method>_seed<S>, methods cei_mp2, graphcare, gsat, protgnn) link each file to the
original and write a binding.json in which the author's repository prefix is replaced by ours and `artifact` points to
the rebuilt trained artifact folder. No hash field is touched. Prints every field that differs, per distinct field.

Usage: python analysis/trained_runs.py <runs root> <new runs root> <repository root> <trained artifact dir>
"""
import json
import os
import sys
from pathlib import Path

OLD_PREFIX = '/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN-fair-v10/'
METHODS = ('cei_mp2', 'graphcare', 'gsat', 'protgnn')


def remap(node, new_prefix, path=''):
    if isinstance(node, dict):
        return {k: remap(v, new_prefix, f'{path}/{k}') for k, v in node.items()}
    if isinstance(node, list):
        return [remap(v, new_prefix, f'{path}[{i}]') for i, v in enumerate(node)]
    if isinstance(node, str) and node.startswith(OLD_PREFIX):
        return new_prefix + node[len(OLD_PREFIX):]
    return node


def flat(node, path=''):
    if isinstance(node, dict):
        return {k2: v2 for k, v in node.items() for k2, v2 in flat(v, f'{path}/{k}').items()}
    if isinstance(node, list):
        return {k2: v2 for i, v in enumerate(node) for k2, v2 in flat(v, f'{path}[{i}]').items()}
    return {path: node}


def main(src, dst, repo, artifact):
    src, dst = Path(src), Path(dst)
    changed, runs = set(), 0
    for run in sorted(src.glob('top*/*_seed*')):
        if run.name.rsplit('_seed', 1)[0] not in METHODS:
            continue
        out = dst / run.parent.name / run.name
        out.mkdir(parents=True, exist_ok=False)
        for item in run.iterdir():
            if item.name != 'binding.json':
                os.symlink(item.resolve(), out / item.name)
        original = json.loads((run / 'binding.json').read_text())
        patched = remap(original, repo.rstrip('/') + '/')
        patched['artifact'] = str(artifact)
        before, after = flat(original), flat(patched)
        changed |= {k for k in before if before[k] != after[k]}
        (out / 'binding.json').write_text(json.dumps(patched, indent=2, sort_keys=True) + '\n')
        runs += 1
    print(runs, 'runs; fields that differ from the originals:')
    for key in sorted(changed):
        print(' ', key)


if __name__ == '__main__':
    main(*sys.argv[1:5])
