import random
from collections import Counter


def select_top_labels(targets, k):
    """Keep the k most frequent TRAIN labels; return (kept ids, remap, dropped counts).

    Frequency is counted on the TRAIN fold only. Ranking classes by their overall
    frequency would let the validation fold decide which classes exist, which is a
    label-distribution leak even though no individual validation row is read.

    Rows outside the kept set are DROPPED, not merged into an "other" class. Merging
    would invent a heterogeneous class that no clinician would recognise and would
    quietly re-inflate the difficulty this option is removing. Ties at the boundary
    are broken by class index so the selection is deterministic.

    **This changes the task, not the model.** A k-class score is not comparable with
    a 30-class score: fewer classes means a higher random floor, a higher majority
    baseline, and the hardest rare classes removed from the denominator of macro-F1.
    Every run records `num_classes` and `dropped_rows` so the two can never be put
    in one table by accident.
    """
    counts = Counter(t for t, split, _ in targets.values() if split == 'train')
    if not counts:
        raise ValueError('No training rows to rank labels by')
    if k < 1 or k > len(counts):
        raise ValueError(f'--top-k-labels {k} exceeds the {len(counts)} training classes')
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:k]
    kept = sorted(ranked)
    remap = {c: i for i, c in enumerate(kept)}
    dropped = Counter()
    filtered = {}
    for sid, (t, split, subject) in targets.items():
        if t in remap:
            filtered[sid] = (remap[t], split, subject)
        else:
            dropped[split] += 1
    return filtered, kept, dict(dropped)


def sample_train_ids(targets, limit, sample_seed):
    """The seeded TRAIN subsample shared by every arm."""
    train_ids = {sid for sid, entry in targets.items() if entry[1] == 'train'}
    if limit is None:
        return train_ids
    rng = random.Random(sample_seed)
    return set(rng.sample(sorted(train_ids), min(limit, len(train_ids))))


def select_dev_ids(targets, train_ids, dev_limit, sample_seed, eligible=None):
    """Patient-disjoint development rows from TRAIN-fold patients outside the sample.

    Epoch and hyper-parameter selection happen on these rows, so the validation fold
    is read once per final run and the test fold stays closed. Excluding every patient
    of the drawn sample stops one patient from informing both fitting and selection.
    """
    if dev_limit is None:
        return frozenset()
    if isinstance(dev_limit, bool) or int(dev_limit) < 1:
        raise ValueError('dev_limit must be a positive integer')
    sampled_subjects = {targets[sid][2] for sid in train_ids}
    candidates = sorted(sid for sid, entry in targets.items()
                        if entry[1] == 'train' and sid not in train_ids
                        and entry[2] not in sampled_subjects
                        and (eligible is None or sid in eligible))
    if len(candidates) < int(dev_limit):
        raise ValueError(f'only {len(candidates)} patient-disjoint TRAIN rows remain for '
                         f'a {int(dev_limit)}-row dev split; lower --dev-limit or '
                         '--train-limit')
    rng = random.Random(f'dev-{sample_seed}')
    return frozenset(rng.sample(candidates, int(dev_limit)))


def split_full_train_dev(targets, dev_limit, sample_seed, eligible=None):
    """fair_v10 full-TRAIN split: (train ids, dev ids), patient-disjoint, dev drawn first.

    The dev split is `dev_limit` rows made of WHOLE TRAIN-fold patients (all their eligible
    rows), drawn by shuffling the sorted patients with random.Random("dev-<sample_seed>") and
    adding every patient whose rows still fit in the remaining dev budget. Train is every other
    TRAIN-fold row whose patient is not in dev (so no dev patient keeps any row in train).
    `targets` must already be top-K filtered. `eligible` (min-prior-visits) restricts which rows
    count toward dev; it does not change which patients are excluded from train.
    """
    if isinstance(dev_limit, bool) or dev_limit is None or int(dev_limit) < 1:
        raise ValueError('dev_limit must be a positive integer')
    dev_limit = int(dev_limit)
    rows_by_subject = {}
    for sid in sorted(targets):
        entry = targets[sid]
        if entry[1] == 'train' and (eligible is None or sid in eligible):
            rows_by_subject.setdefault(entry[2], []).append(sid)
    subjects = sorted(rows_by_subject)
    random.Random(f'dev-{sample_seed}').shuffle(subjects)
    dev, dev_subjects = [], set()
    for subject in subjects:
        remaining = dev_limit - len(dev)
        if remaining == 0:
            break
        rows = rows_by_subject[subject]
        if len(rows) <= remaining:
            dev.extend(rows)
            dev_subjects.add(subject)
    if len(dev) != dev_limit:
        raise ValueError(f'only {len(dev)} dev rows can be drawn from whole TRAIN patients; '
                         f'lower --dev-limit')
    train_ids = {sid for sid, entry in targets.items()
                 if entry[1] == 'train' and entry[2] not in dev_subjects}
    return train_ids, frozenset(dev)


def resolve_train_dev_ids(targets, limit, dev_limit, sample_seed, eligible=None):
    """(train ids, dev ids) of one run. Numeric `limit`: the historical sample + outside-dev draw.

    `limit=None` with a `dev_limit` is the fair_v10 full-TRAIN rule (`split_full_train_dev`);
    `limit=None` and `dev_limit=None` is the fair_v10t rule: train = EVERY TRAIN-fold row of
    `targets` (already top-K filtered), empty dev set (checkpoint selection uses validation).
    """
    if limit is None and dev_limit is not None:
        return split_full_train_dev(targets, dev_limit, sample_seed, eligible)
    if limit is None:
        return ({sid for sid, entry in targets.items() if entry[1] == 'train'}, frozenset())
    train_ids = sample_train_ids(targets, limit, sample_seed)
    return train_ids, select_dev_ids(targets, train_ids, dev_limit, sample_seed, eligible)
