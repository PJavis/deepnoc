"""
Splits that don't leak.

`train_test_split_alternating` (the original) takes every second profile.
PROVEDIt re-injects the same physical mixture at multiple times / amplifications,
so adjacent rows are often siblings of the same biological sample. Splitting
alternately therefore puts near-identical profiles in train and test and inflates
test accuracy.

Two replacements:

    `stratified_split`
        Random split stratified by NoC. Honest single-fold split.

    `grouped_stratified_split`
        Random split stratified by NoC AND grouped by a "pedigree" key derived
        from the sample name. Two profiles that share the pedigree key end up
        on the same side of the split. Use this whenever names are available.

Both return (X_train, X_test, y_train, y_test, names_train, names_test).
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit


def _pedigree_key(name: str) -> str:
    """
    Extract a stable group key from a PROVEDIt sample name.

    PROVEDIt names look like
        RD14-0003_GF_25sec_GM_SE33F_2-5P:A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_...
        RD14-0003_GF_25sec_GM_SE33F_1P:A02_RD14-0003-15d2U60-0.25GF-Q4.5_...

    After the colon comes a per-sample suffix shaped like
        <well>_RD<x>-<y>-<contributor_pedigree>-<conditions>...

    We want `<contributor_pedigree>` because every replicate / re-injection of
    the same biological mixture shares it, and putting any of them on
    different sides of the split would leak information.

    Strategy:
      1. Take the part after ':' (or fall back to whole name).
      2. Strip the leading well + 'RD<digits>-<digits>-' prefix.
      3. The pedigree is everything up to the first '-M' or '-S' or '-Q'
         (those letters mark the start of the experimental-condition tags).
    """
    suffix = name.split(":", 1)[1] if ":" in name else name
    suffix = re.sub(r"^[A-Z]\d{2}_", "", suffix)              # drop well "A02_"
    suffix = re.sub(r"^RD\d+-\d+-", "", suffix)               # drop "RD14-0003-"
    m = re.match(r"([^-]+(?:-[^-]+)*?)(?=-(?:M|S|Q|\d+\.\d))", suffix)
    if m:
        return m.group(1)
    # Fallback: take up to the first underscore (well-only is still better than nothing).
    return suffix.split("_", 1)[0] if "_" in suffix else suffix


def stratified_split(X, y, names, test_size: float = 0.25, seed: int = 42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    return (X[tr], X[te], y[tr], y[te],
            [names[i] for i in tr], [names[i] for i in te])


def grouped_stratified_split(X, y, names, test_size: float = 0.25, seed: int = 42):
    """
    Group-aware stratified split. Implementation: GroupShuffleSplit on pedigree
    keys, with a few retries until each NoC class has at least one sample on
    both sides. Pure stratification + grouping is over-constrained when a
    pedigree group only contains one NoC, which is the common case here, so
    the retries are usually a no-op.
    """
    groups = np.array([_pedigree_key(n) for n in names])
    rng = np.random.default_rng(seed)
    for attempt in range(20):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=int(rng.integers(0, 2**31 - 1)))
        (tr, te), = gss.split(np.zeros(len(y)), y, groups=groups)
        if (set(np.unique(y[tr])) >= set(np.unique(y))
                and set(np.unique(y[te])) >= set(np.unique(y))):
            break
    return (X[tr], X[te], y[tr], y[te],
            [names[i] for i in tr], [names[i] for i in te])
