"""Graph hash check: relate a rebuilt final graph to the file the final runs were trained on.

The 45 final runs (clinical_runs_fair_v10t_20261011) were trained on the graph built on 2026-10-10. Their
binding.json files record graphs.jsonl sha256 53c92ea0... (6,413,118,495 bytes). Before the repository was
published, one provenance label was renamed in the builder: the `reviewed_by` value of the source-checked
knowledge rows changed from OLD_LABEL to NEW_LABEL. Nothing else changed. A rebuild with this repository therefore
gives graphs.jsonl 3aa1dc65... (6,435,138,687 bytes; the value in the README table), and thresholds.csv differs in
the same column. visit_membership.jsonl is unaffected.

This script hashes the rebuilt files twice: as they are, and with NEW_LABEL swapped back to OLD_LABEL in memory.
The swapped hashes must equal the training hashes. manifest.json cannot be compared this way: it records absolute
paths and the builder code hashes of the build machine.

Usage: python -m comparison.final.graph_hash_check [--graph-dir DIR] [--binding RUN/binding.json]
Opens every input read-only, writes nothing and prints hashes and byte counts only (no record contents).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_GRAPH_DIR = 'comparison/standardized/event_inputs/clinical_graph_L4_plus60_v5c_20261009'
OLD_LABEL = b'hermes-source-check-not-clinical-review'
NEW_LABEL = b'automated-source-check-not-clinical-review'

# file name -> (sha256 as rebuilt by this repository, sha256 of the file the final runs were trained on, bytes)
EXPECTED = {
    'graphs.jsonl': ('3aa1dc658e7153faf045bb70f43ca065b222e362ea0c86bcb4a60a35666858dd',
                     '53c92ea09159274528b2cb2ca454ad89ddb8064721007e29821a570f1ddf91ad', 6413118495),
    'thresholds.csv': ('3571a87d64869925a044276d138c38ced73d19628ae2fba241146bb83f774b12',
                       '0113be57055febb5dc26b1b3b8b8fe9cf1d44e802fb3477f4485339b355d2b04', 78820),
    'visit_membership.jsonl': ('9e2399f39c57ef0fb8831eccd8f118e5315a3cb1fd781f1161685881300090ed',
                               '9e2399f39c57ef0fb8831eccd8f118e5315a3cb1fd781f1161685881300090ed', 190700935),
}
BINDING_FIELDS = {'graphs.jsonl': ('artifact_graphs_sha256', 'graph_sha256'),
                  'visit_membership.jsonl': ('artifact_visit_membership_sha256',)}


def hashes(path):
    """sha256 of the file as it is and with NEW_LABEL swapped back to OLD_LABEL, plus the swapped byte count."""
    as_is, swapped, size = hashlib.sha256(), hashlib.sha256(), 0
    with open(path, 'rb') as handle:
        for line in handle:
            as_is.update(line)
            line = line.replace(NEW_LABEL, OLD_LABEL)
            swapped.update(line)
            size += len(line)
    return as_is.hexdigest(), swapped.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description='Relate a rebuilt final graph to the trained graph (read-only).')
    parser.add_argument('--graph-dir', default=DEFAULT_GRAPH_DIR)
    parser.add_argument('--binding', help='binding.json of a final run; its recorded hashes become the targets')
    args = parser.parse_args()

    trained = {name: values[1] for name, values in EXPECTED.items()}
    if args.binding:
        binding = json.loads(Path(args.binding).read_text())
        for name, fields in BINDING_FIELDS.items():
            recorded = {binding.get(field) for field in fields} - {None}
            if len(recorded) != 1:
                sys.exit(f'{args.binding}: expected one value for {fields}, found {sorted(recorded)}')
            trained[name] = recorded.pop()

    ok = True
    for name, (rebuilt_sha, _, trained_bytes) in EXPECTED.items():
        path = Path(args.graph_dir) / name
        if not path.is_file():
            print(f'  [FAIL] {name}: missing in {args.graph_dir}')
            ok = False
            continue
        as_is, swapped, size = hashes(path)
        print(f'{name}')
        print(f'  as rebuilt          {as_is}  ({path.stat().st_size:,} bytes)'
              f'  {"= repository rebuild" if as_is == rebuilt_sha else "!= repository rebuild " + rebuilt_sha[:12]}')
        match = swapped == trained[name] and size == trained_bytes
        ok = ok and match
        print(f'  [{"OK" if match else "FAIL"}] label swapped back {swapped}  ({size:,} bytes)'
              f'  {"=" if match else "!="} trained file {trained[name][:12]}...')
    print('All files reproduce the trained inputs.' if ok else 'Mismatch: see the FAIL lines.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
