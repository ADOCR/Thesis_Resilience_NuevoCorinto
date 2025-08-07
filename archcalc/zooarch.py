"""Funciones de zooarqueología y tafonomía"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from numpy.typing import ArrayLike


def nisp(counts: ArrayLike) -> int:
    return int(np.sum(counts))


def mne(elements: Dict[str, int]) -> Dict[str, int]:
    return elements


def mni(elements: Dict[str, int], per_individual: Dict[str, int]) -> int:
    ratios = []
    for elem, count in elements.items():
        if elem in per_individual:
            ratios.append(count / per_individual[elem])
    return int(np.max(ratios)) if ratios else 0


def fragmentation_index(nisp_value: int, mne_value: int) -> float:
    return nisp_value / mne_value


def percentage(partial: int, total: int) -> float:
    return (partial / total) * 100


def mortality_profile(ages: ArrayLike, bins: Iterable[int]) -> np.ndarray:
    ages = np.asarray(ages)
    hist, _ = np.histogram(ages, bins=bins)
    return hist


def survivorship_curve(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    surv = {}
    cumulative = 0
    for elem, count in sorted(counts.items(), key=lambda x: x[0]):
        cumulative += count
        surv[elem] = 1 - cumulative / total
    return surv


def estimated_biomass(nisp_value: int, meat_weight: float) -> float:
    return nisp_value * meat_weight
