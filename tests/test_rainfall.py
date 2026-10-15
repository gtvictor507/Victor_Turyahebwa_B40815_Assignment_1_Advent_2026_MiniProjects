import numpy as np
from scipy.spatial.distance import cosine

from src.rainfall import CropRule, Region, compare, cosine_similarity


def test_region_months_and_totals():
    region = Region('Example', list(range(1, 13)))
    assert region.annual_total() == 78
    assert region.driest() == 'Jan' and region.wettest() == 'Dec'


def test_cosine_matches_scipy():
    a = np.array([1., 2., 3.])
    b = np.array([2., 0., 4.])
    assert np.isclose(cosine_similarity(a, b), 1-cosine(a, b))


def test_crop_classes_and_invalid_values():
    crop = CropRule('Bean', 100, 167)
    assert crop.classify(50) == 'Drought risk'
    assert crop.classify(150) == 'Suitable'
    assert crop.classify(200) == 'Waterlogging risk'
    try:
        Region('Wrong', [-1]*12)
        assert False, 'Negative rainfall should fail'
    except ValueError:
        pass


def test_comparison_of_scaled_profiles():
    a = Region('A', [10]*6 + [20]*6)
    b = Region('B', [20]*6 + [40]*6)
    result = compare(a, b)
    assert np.isclose(result['cosine'], 1)
    assert np.isclose(result['pearson'], 1)
    assert result['euclidean_mm'] > 0
