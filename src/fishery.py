"""Harvested logistic fish stock and price-based revenue risk."""

import statistics

import numpy as np


class FishStock:
    """Weekly stock in tonnes; harvest is h times the starting stock each week."""

    def __init__(self, initial=4000.0, growth=0.4, capacity=10000.0, harvest_rate=0.1):
        if initial < 0 or growth < 0 or capacity <= 0 or not 0 <= harvest_rate <= 1:
            raise ValueError('Stock, growth, capacity or harvest rate is invalid')
        self.initial, self.growth, self.capacity = initial, growth, capacity
        self.harvest_rate = harvest_rate

    def simulate(self, weeks: int) -> tuple:
        if weeks < 1:
            raise ValueError('Weeks must be positive')
        stock = [float(self.initial)]
        harvests = []
        for _ in range(weeks):
            current = stock[-1]
            catch = self.harvest_rate * current
            growth = self.growth * current * (1-current/self.capacity)
            harvests.append(catch)
            stock.append(max(0, current + growth-catch))
        return np.array(stock), np.array(harvests)

    def maximum_sustainable_yield(self) -> float:
        return self.growth*self.capacity/4


class PriceModel:
    """Bounded seeded weekly random-walk price in UGX per kg."""

    def __init__(self, start=12000, lower=9000, upper=16000, noise=250):
        if not 0 < lower <= start <= upper or noise < 0:
            raise ValueError('Invalid price bounds or volatility')
        self.start, self.lower, self.upper, self.noise = start, lower, upper, noise

    def simulate(self, weeks: int, rng: np.random.Generator) -> np.ndarray:
        if weeks < 1:
            raise ValueError('Weeks must be positive')
        prices, last = [], self.start
        for _ in range(weeks):
            last = float(np.clip(last + rng.normal(0, self.noise), self.lower, self.upper))
            prices.append(last)
        return np.array(prices)


class RiskAssessor:
    """Classify weekly revenue dispersion using its dimensionless CV."""

    def describe(self, revenue) -> dict:
        data = list(map(float, revenue))
        if len(data) < 2 or statistics.mean(data) <= 0:
            raise ValueError('At least two values with positive mean are required')
        mean = statistics.mean(data)
        return {'mean': mean, 'median': statistics.median(data),
                'variance': statistics.variance(data), 'std_dev': statistics.stdev(data),
                'CV': statistics.stdev(data)/mean}

    def classify(self, cv: float) -> str:
        if cv < 0:
            raise ValueError('CV cannot be negative')
        return 'low' if cv < 0.15 else 'moderate' if cv < 0.30 else 'high'


def annual_revenue_samples(harvest_tonnes, price_model, repeats=1200, seed=42):
    """One RNG stream generates many independent price paths; tonnes -> kg."""
    rng = np.random.default_rng(seed)
    return np.array([np.sum(harvest_tonnes*1000*price_model.simulate(
        len(harvest_tonnes), rng)) for _ in range(repeats)])
