"""Estadística de conjuntos arqueológicos"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rarefaction(counts: ArrayLike, sample_size: int) -> float:
    counts = np.asarray(counts)
    n = counts.sum()
    if sample_size > n:
        raise ValueError("sample_size mayor que el total")
    probs = 1 - np.exp(np.sum(np.log(1 - np.arange(sample_size) / n)))
    return probs


def shannon(counts: ArrayLike) -> float:
    counts = np.asarray(counts)
    p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log(p))


def simpson(counts: ArrayLike) -> float:
    counts = np.asarray(counts)
    p = counts / counts.sum()
    return 1 - np.sum(p**2)


def hill_number(counts: ArrayLike, q: float) -> float:
    counts = np.asarray(counts)
    p = counts / counts.sum()
    if q == 1:
        return np.exp(-np.sum(p * np.log(p)))
    return (np.sum(p**q)) ** (1 / (1 - q))


def evenness(counts: ArrayLike) -> float:
    s = np.count_nonzero(counts)
    return shannon(counts) / np.log(s)


def chao1(counts: ArrayLike) -> float:
    counts = np.asarray(counts)
    f1 = np.sum(counts == 1)
    f2 = np.sum(counts == 2)
    s_obs = np.count_nonzero(counts)
    if f2 == 0:
        return s_obs + f1 * (f1 - 1) / 2
    return s_obs + (f1**2) / (2 * f2)


def ace(counts: ArrayLike, threshold: int = 10) -> float:
    counts = np.asarray(counts)
    rare = counts[counts <= threshold]
    abundant = counts[counts > threshold]
    s_rare = np.count_nonzero(rare)
    s_abund = np.count_nonzero(abundant)
    n_rare = rare.sum()
    c_ace = 1 - np.sum(rare == 1) / n_rare
    gamma_sq = s_rare * np.var(rare, ddof=1) / (c_ace * n_rare ** 2)
    return s_abund + s_rare / c_ace + (np.sum(rare == 1) * gamma_sq)


def bootstrap_diff(data1: ArrayLike, data2: ArrayLike, n: int = 1000) -> np.ndarray:
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diffs = []
    for _ in range(n):
        s1 = np.random.choice(data1, size=len(data1), replace=True)
        s2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(s1.mean() - s2.mean())
    return np.array(diffs)


def simple_permanova(distance_matrix: ArrayLike, groups: ArrayLike, permutations: int = 999):
    try:
        from skbio.stats.distance import permanova, DistanceMatrix
    except Exception as exc:  # pragma: no cover
        raise ImportError("scikit-bio es necesario para PERMANOVA") from exc
    dm = DistanceMatrix(distance_matrix)
    return permanova(dm, groups, permutations=permutations)


def monte_carlo_proportion(success: int, total: int, n: int = 1000) -> np.ndarray:
    p = success / total
    return np.random.binomial(total, p, size=n) / total


def propagate_error(values: ArrayLike, errors: ArrayLike) -> float:
    values = np.asarray(values)
    errors = np.asarray(errors)
    return np.sqrt(np.sum(errors**2)) / len(values)
