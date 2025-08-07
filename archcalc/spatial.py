"""Funciones espaciales y de prospección"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde


def utm_to_latlon(easting: float, northing: float, zone: int, northern: bool = True) -> Tuple[float, float]:
    transformer = Transformer.from_crs(f"epsg:326{zone}" if northern else f"epsg:327{zone}", "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


def latlon_to_utm(lat: float, lon: float, zone: int, northern: bool = True) -> Tuple[float, float]:
    transformer = Transformer.from_crs("epsg:4326", f"epsg:326{zone}" if northern else f"epsg:327{zone}", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    line = LineString([p1, p2])
    return line.length


def area(coords: Iterable[Tuple[float, float]]) -> float:
    poly = Polygon(coords)
    return poly.area


def slope(elev1: float, elev2: float, horizontal: float) -> float:
    return (elev2 - elev1) / horizontal


def buffer(point: Tuple[float, float], radius: float) -> Polygon:
    return Point(point).buffer(radius)


def route_length(coords: Iterable[Tuple[float, float]]) -> float:
    line = LineString(list(coords))
    return line.length


def density_kde(points: np.ndarray, bandwidth: float = None) -> gaussian_kde:
    return gaussian_kde(points.T, bw_method=bandwidth)


def nearest_neighbor(points: np.ndarray) -> float:
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    return np.mean(dists[:, 1])


def ripleys_k(points: np.ndarray, radii: Iterable[float], area_polygon: Polygon) -> np.ndarray:
    tree = KDTree(points)
    area = area_polygon.area
    k_values = []
    for r in radii:
        counts = tree.query_ball_point(points, r)
        n = sum(len(c) - 1 for c in counts)
        lambda_hat = len(points) / area
        k = n / (len(points) * lambda_hat)
        k_values.append(k)
    return np.array(k_values)


def sampling_effort(surveyed_area: float, total_area: float) -> float:
    return surveyed_area / total_area
