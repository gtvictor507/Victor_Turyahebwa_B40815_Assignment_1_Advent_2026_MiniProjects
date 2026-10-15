import numpy as np

from src.population import DistrictPopulation, ExponentialGrowth, FibonacciRatio, LinearTrend, error_metrics


def test_population_growth_and_length():
    district = DistrictPopulation('Test', [2020, 2021, 2022], [100, 110, 121])
    assert len(district) == 3
    assert np.isclose(district.cagr(), .10)
    assert 'Test' in repr(district)


def test_linear_forecast_and_metrics():
    model = LinearTrend().fit([100, 110, 120])
    assert np.isclose(model.predict(1)[0], 130)
    assert error_metrics([2, 4], [1, 3])['MAE'] == 1


def test_invalid_population_and_short_series():
    try:
        DistrictPopulation('Bad', [2020, 2021], [100, -1])
        assert False, 'Negative population should fail'
    except ValueError:
        pass
    try:
        FibonacciRatio().fit([1])
        assert False, 'One observation should fail'
    except ValueError:
        pass


def test_exponential_forecast():
    assert np.isclose(ExponentialGrowth().fit([100, 110]).predict(1)[0], 121)
