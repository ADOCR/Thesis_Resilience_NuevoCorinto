"""Funciones paleoetnobotánicas"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def pollen_percentage(count: int, total: int) -> float:
    return (count / total) * 100


def lycopodium_concentration(count_pollen: int, count_lyco: int, tablet_conc: float) -> float:
    return count_pollen * tablet_conc / count_lyco


def concentration(count: int, weight: float = None, volume: float = None) -> float:
    if weight is not None:
        return count / weight
    if volume is not None:
        return count / volume
    raise ValueError("se requiere peso o volumen")


def flux(concentration: float, sedimentation_rate: float) -> float:
    return concentration * sedimentation_rate


def accumulation_curve(counts: ArrayLike) -> np.ndarray:
    counts = np.asarray(counts)
    return np.cumsum(counts)


def rarefaction(counts: ArrayLike, sample_size: int) -> float:
    counts = np.asarray(counts)
    n = counts.sum()
    if sample_size > n:
        raise ValueError("sample_size mayor que total")
    probs = 1 - np.exp(np.sum(np.log(1 - np.arange(sample_size) / n)))
    return probs


def contamination_table(blanks: ArrayLike, samples: ArrayLike) -> pd.DataFrame:
    return pd.DataFrame({"blanks": blanks, "samples": samples})


def normalize_phytoliths(area: float, fragments: int) -> float:
    return fragments / area


def microcharcoal_rate(volume: float, fragments: int) -> float:
    return fragments / volume
