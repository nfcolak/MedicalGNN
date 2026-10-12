"""ICD-9 -> ICD-10 crosswalk v2: GEM + tracked tie-break overrides (policy gem_override_v1_stem98short).

The v3/v4 crosswalk (icd_mapping.load_icd_map) breaks ties between equally ranked GEM
candidates with pandas quicksort order, which is arbitrary (4019 -> I16.9 crisis, 99859 ->
K68.11, 427.89 -> R00.1). This module leaves that module untouched and adds a reviewed
replacement used only by the v5 builds and primary_dx_v4 labels:

1. icd9_tiebreak_overrides_v1.csv (columns icd9, icd10, reason) wins. icd10 must be a GEM
   candidate (no_map == 0) of that ICD-9 code. An empty icd10 means no defensible single
   target: the crosswalk value is None, the graph keeps dx:icd9:<code> and the label is -1.
2. Codes without an override whose best-ranked (lowest `approximate`) candidates still
   differ use the fallback below. Codes with a single best-ranked candidate keep it, so
   every unambiguous code maps exactly as in icd_mapping.load_icd_map.

Fallback (`fallback_pick`): compare candidates on the stem (drop the 7th character, then
trailing X placeholders): stem ending in 9 > ending in 8 > shortest stem > alphabetical on
the full code. ICD-9 E-codes first prefer targets whose 7th character is A (initial
encounter).
"""
import hashlib
from pathlib import Path

import pandas as pd

ICD_MAPPING_POLICY_V2 = 'gem_override_v1_stem98short'
OVERRIDES_PATH = Path(__file__).with_name('icd9_tiebreak_overrides_v1.csv')


def code_stem(icd10):
    """ICD-10 code without its 7th character (when present) and trailing X placeholders."""
    stem = icd10[:6] if len(icd10) == 7 else icd10
    return stem.rstrip('X') or stem


def fallback_pick(icd9, candidates):
    """Deterministic pick among tied ICD-10 candidates of one ICD-9 code (see module doc)."""
    is_e = icd9.upper().startswith('E')

    def key(code):
        stem = code_stem(code)
        return (0 if (not is_e or (len(code) == 7 and code[6] == 'A')) else 1,
                0 if stem.endswith('9') else 1,
                0 if stem.endswith('8') else 1,
                len(stem), code)
    return min((c.upper() for c in candidates), key=key)


def overrides_sha256(path=OVERRIDES_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_overrides(path=OVERRIDES_PATH):
    """Return {icd9: icd10 or None}; None = no defensible single target."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ['icd9', 'icd10', 'reason']:
        raise ValueError('ICD override table needs exactly the columns icd9,icd10,reason')
    codes = frame.icd9.str.strip().str.upper()
    if (codes == '').any() or codes.duplicated().any():
        raise ValueError('ICD override table has empty or duplicated icd9 codes')
    return {code: (target.strip().upper() or None) for code, target in zip(codes, frame.icd10)}


def load_icd_map_v2(path, overrides_path=OVERRIDES_PATH):
    """Return (crosswalk, approximate_codes) like icd_mapping.load_icd_map.

    crosswalk maps icd9 -> icd10, or None for override rows without a target.
    approximate_codes holds the ICD-9 codes whose chosen GEM row is flagged approximate.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {'icd9_code', 'icd10_code', 'approximate', 'no_map'}.issubset(frame.columns):
        raise ValueError('ICD mapping is missing required columns')
    frame['approximate'] = pd.to_numeric(frame.approximate, errors='raise')
    rows = {}
    for row in frame[frame.no_map == '0'].itertuples(index=False):
        code, target = row.icd9_code.upper(), row.icd10_code.upper()
        if not code or not target:
            raise ValueError('Empty code in selected ICD mapping')
        best = rows.setdefault(code, {})
        best[target] = min(row.approximate, best.get(target, row.approximate))
    if not rows:
        raise ValueError('Empty ICD-9 to ICD-10 mapping')
    overrides = read_overrides(overrides_path)
    unknown = sorted(code for code in overrides if code not in rows)
    if unknown:
        raise ValueError('ICD overrides for codes without GEM candidates: ' + ', '.join(unknown))
    mapping, approximate = {}, set()
    for code, targets in rows.items():
        if code in overrides:
            target = overrides[code]
            if target is not None and target not in targets:
                raise ValueError('ICD override %s -> %s is not a GEM candidate' % (code, target))
        else:
            top = min(targets.values())
            tied = sorted(t for t, rank in targets.items() if rank == top)
            target = tied[0] if len(tied) == 1 else fallback_pick(code, tied)
        mapping[code] = target
        if target is not None and targets[target] == 1:
            approximate.add(code)
    return mapping, approximate


def normalize_icd10_v2(version, code, mapping):
    """Map a diagnosis to ICD-10; None for empty, unmapped or no-target ICD-9 codes."""
    code = code.strip().upper()
    if version not in ('9', '10'):
        raise ValueError('Unsupported ICD version: ' + version)
    if not code:
        return None
    return code if version == '10' else mapping.get(code)
