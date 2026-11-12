"""Statistics helpers."""
from __future__ import annotations

from itertools import combinations

from scipy.stats import fisher_exact


def flip_test(flips: int, n: int, noise_flips: int, noise_n: int) -> float:
    """One-sided Fisher exact test: is the flip rate under perturbation higher than the noise floor?"""
    if n == 0:
        return 1.0
    _, p = fisher_exact([[flips, n - flips], [noise_flips, noise_n - noise_flips]], alternative="greater")
    return float(p)


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * m / rank)
        adj[i] = prev
    return adj


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets) -> float:
    pairs = list(combinations(sets, 2))
    return 1.0 if not pairs else sum(jaccard(x, y) for x, y in pairs) / len(pairs)


def precision_recall(cited, determinant) -> dict:
    cited, determinant = set(cited), set(determinant)
    tp = cited & determinant
    p = len(tp) / len(cited) if cited else (1.0 if not determinant else 0.0)
    r = len(tp) / len(determinant) if determinant else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "true_positive": sorted(tp),
            "cited_not_determinant": sorted(cited - determinant), "determinant_not_cited": sorted(determinant - cited)}
