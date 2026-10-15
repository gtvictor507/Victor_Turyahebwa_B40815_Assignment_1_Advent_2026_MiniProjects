import numpy as np

from src.taxi import (ExponentialSmoothing, LinearTrend, MovingAverage,
                      Route, equilibrium, vehicles_needed, walk_forward)


def test_route_revenue_and_equilibrium():
    route = Route('Test', [1, 2, 3, 4], 2000)
    assert route.total_revenue() == 20000
    assert np.allclose(equilibrium(), [2200, 76])


def test_forecasts_and_walk_forward():
    assert MovingAverage().fit([1, 2, 3]).predict() == 2
    assert np.isclose(LinearTrend().fit([1, 2, 3]).predict(), 4)
    mae, predictions = walk_forward([1, 2, 3, 4, 5], LinearTrend)
    assert np.isclose(mae, 0) and len(predictions) == 2


def test_fleet_and_alpha_validation():
    assert vehicles_needed(112) == 2
    assert vehicles_needed(0) == 0
    try:
        ExponentialSmoothing(0)
        assert False, 'Zero alpha must fail'
    except ValueError:
        pass


def test_negative_passengers_rejected():
    try:
        Route('Bad', [-1, 1, 1, 1], 100)
        assert False, 'Negative counts must fail'
    except ValueError:
        pass
