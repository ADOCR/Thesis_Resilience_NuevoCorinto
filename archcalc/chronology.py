"""Funciones de fechamiento y cronología"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CalibratedDate:
    """Resultado de una calibración simplificada"""

    cal_bp: np.ndarray
    pdf: np.ndarray

    def hpd(self, level: float = 0.68) -> List[Tuple[float, float]]:
        return hpd_intervals(self.cal_bp, self.pdf, level)


def bp_to_cal(bp: float) -> int:
    """Convierte años BP a calendario (BCE negativo)."""
    year = 1950 - bp
    return int(year)


def cal_to_bp(year: int) -> int:
    """Convierte año calendario a BP."""
    return int(1950 - year)


def combine_radiocarbon(dates: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    """Combina fechas 14C usando medias ponderadas."""
    ages = np.array([d[0] for d in dates])
    sigmas = np.array([d[1] for d in dates])
    weights = 1 / sigmas**2
    mean = np.sum(weights * ages) / np.sum(weights)
    sigma = np.sqrt(1 / np.sum(weights))
    return mean, sigma


def calibrate_radiocarbon(age_bp: float, error: float, curve: pd.DataFrame, delta_r: float = 0) -> CalibratedDate:
    """Calibración 14C simplificada."""
    curve = curve.copy()
    curve["c14_bp"] = curve["c14_bp"] + delta_r
    diff = age_bp - curve["c14_bp"]
    total_sigma = np.sqrt(error**2 + curve["sigma"]**2)
    pdf = np.exp(-0.5 * (diff / total_sigma) ** 2)
    pdf /= np.sum(pdf)
    return CalibratedDate(curve["cal_bp"].to_numpy(), pdf.to_numpy())


def hpd_intervals(x: np.ndarray, pdf: np.ndarray, level: float = 0.68) -> List[Tuple[float, float]]:
    order = np.argsort(pdf)[::-1]
    pdf_sorted = pdf[order]
    x_sorted = x[order]
    cumsum = np.cumsum(pdf_sorted)
    mask = cumsum <= level
    idx = order[mask]
    selected = x[idx]
    selected.sort()
    intervals = []
    start = selected[0]
    prev = selected[0]
    for val in selected[1:]:
        if val - prev > 1:
            intervals.append((start, prev))
            start = val
        prev = val
    intervals.append((start, prev))
    return intervals


def phase_summary(dates: Iterable[CalibratedDate]) -> Tuple[float, float]:
    all_cal = np.concatenate([d.cal_bp for d in dates])
    return float(np.min(all_cal)), float(np.max(all_cal))


def report_metadata(labcode: str, d13c: float, curve: str, reference: str) -> str:
    return f"LabCode: {labcode}\nδ13C: {d13c}‰\nCurva: {curve}\nReferencia: {reference}"
