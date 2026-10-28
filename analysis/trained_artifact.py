"""Write the trained graph inputs from the local rebuild: swap the renamed provenance label back.

Reads the rebuilt artifact folder (read-only), writes graphs.jsonl and thresholds.csv with NEW_LABEL replaced by
OLD_LABEL into a new folder, links visit_membership.jsonl, copies the trained manifest.json, and checks every sha256
against the values the final runs recorded. Refuses to write into an existing folder; exits 1 if any hash differs.

The final runs were trained on the graph built before the `reviewed_by` provenance label was renamed (README of the
`fix_hash` branch of nfcolak/MedicalGNN). manifest.json cannot be rebuilt (build-machine paths and code hashes) and is
copied from the author's trained artifact.

Usage: python analysis/trained_artifact.py <rebuilt artifact dir> <new trained dir> <trained manifest.json>
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

OLD_LABEL = b'hermes-source-check-not-clinical-review'
NEW_LABEL = b'automated-source-check-not-clinical-review'
EXPECTED = {
    'graphs.jsonl': '53c92ea09159274528b2cb2ca454ad89ddb8064721007e29821a570f1ddf91ad',
    'thresholds.csv': '0113be57055febb5dc26b1b3b8b8fe9cf1d44e802fb3477f4485339b355d2b04',
    'visit_membership.jsonl': '9e2399f39c57ef0fb8831eccd8f118e5315a3cb1fd781f1161685881300090ed',
    'manifest.json': 'e5d8028ac68e31658cec6cc6588064ded23dec9aa41832597ef9cb067ffff34b',
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def swap_file(src, dst):
    with open(src, 'rb') as source, open(dst, 'xb') as target:
        for line in source:
            target.write(line.replace(NEW_LABEL, OLD_LABEL))


def main(rebuilt, trained, manifest):
    rebuilt, trained, manifest = Path(rebuilt), Path(trained), Path(manifest)
    trained.mkdir(parents=True, exist_ok=False)
    for name in ('graphs.jsonl', 'thresholds.csv'):
        swap_file(rebuilt / name, trained / name)
    os.symlink((rebuilt / 'visit_membership.jsonl').resolve(), trained / 'visit_membership.jsonl')
    shutil.copyfile(manifest, trained / 'manifest.json')
    ok = True
    for name, expected in EXPECTED.items():
        got = sha256_of(trained / name)
        ok = ok and got == expected
        print(f'[{"OK" if got == expected else "FAIL"}] {name} {got[:12]} (expected {expected[:12]}) '
              f'{(trained / name).stat().st_size:,} bytes')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
