import numpy as np

from archcalc import chronology, spatial, assemblage, paleo, zooarch


def test_bp_conversion():
    assert chronology.bp_to_cal(50) == 1900
    assert chronology.cal_to_bp(1900) == 50


def test_combine():
    mean, sigma = chronology.combine_radiocarbon([(1000, 30), (1010, 25)])
    assert round(mean, 1) == 1005.5
    assert sigma > 0


def test_spatial_conversion():
    e, n = spatial.latlon_to_utm(0, 3, zone=31)
    lat, lon = spatial.utm_to_latlon(e, n, zone=31)
    assert abs(lat) < 1e-6
    assert abs(lon - 3) < 1e-6


def test_shannon():
    counts = [10, 10, 10]
    assert round(assemblage.shannon(counts), 3) == round(np.log(3), 3)


def test_pollen_percentage():
    assert paleo.pollen_percentage(20, 100) == 20


def test_nisp():
    assert zooarch.nisp([1, 2, 3]) == 6
