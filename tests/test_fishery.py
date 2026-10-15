import numpy as np

from src.fishery import FishStock, PriceModel, RiskAssessor, annual_revenue_samples


def test_first_step_logistic_harvest():
    stock, catch = FishStock().simulate(1)
    assert np.isclose(catch[0], 400)
    assert np.isclose(stock[1], 4000+0.4*4000*0.6-400)


def test_reproducible_bounded_prices():
    model = PriceModel()
    a = model.simulate(20, np.random.default_rng(42))
    b = model.simulate(20, np.random.default_rng(42))
    assert np.array_equal(a, b)
    assert np.all((9000 <= a) & (a <= 16000))


def test_var_and_risk_threshold():
    assessor = RiskAssessor()
    assert assessor.classify(0.20) == 'moderate'
    samples = annual_revenue_samples(np.array([1., 1.]), PriceModel(), repeats=1000)
    assert len(samples) == 1000 and np.percentile(samples, 5) > 0


def test_invalid_harvest_rate():
    try:
        FishStock(harvest_rate=-.1)
        assert False, 'Invalid harvest should fail'
    except ValueError:
        pass
