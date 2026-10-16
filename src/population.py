"""District population data and three transparent forecasting methods."""

import statistics
from abc import ABC, abstractmethod

import numpy as np


class DistrictPopulation:
    """A district's yearly population estimates, measured in thousands."""

    def __init__(self, name: str, years, populations):
        self.name = name
        # Arrays make later growth and forecasting calculations element-wise.
        self.years = np.asarray(years, dtype=int)
        self.populations = np.asarray(populations, dtype=float)
        if self.years.ndim != 1 or self.populations.ndim != 1:
            raise ValueError('Years and populations must be one-dimensional')
        if len(self.years) != len(self.populations) or len(self.years) < 2:
            raise ValueError('At least two matching years and populations are required')
        if not np.all(np.diff(self.years) == 1):
            raise ValueError('Years must be consecutive and increasing')
        if np.any(~np.isfinite(self.populations)) or np.any(self.populations < 0):
            raise ValueError('Populations must be finite and non-negative')

    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return f'DistrictPopulation({self.name!r}, {self.years[0]}-{self.years[-1]})'

    def describe(self) -> dict:
        """Return sample variance and sample SD (ddof=1)."""
        data = self.populations.tolist()
        # statistics.variance uses n-1, unlike the default np.var (which uses n).
        return {'mean': statistics.mean(data), 'median': statistics.median(data),
                'variance': statistics.variance(data), 'std_dev': statistics.stdev(data)}

    def growth_rates(self) -> np.ndarray:
        """Year-on-year proportional changes."""
        if np.any(self.populations[:-1] <= 0):
            raise ValueError('Growth requires positive starting populations')
        # Each year is compared with the immediately preceding year.
        return self.populations[1:] / self.populations[:-1] - 1

    def cagr(self) -> float:
        if self.populations[0] <= 0:
            raise ValueError('CAGR requires a positive starting population')
        # Ten annual observations contain nine one-year growth intervals.
        return (self.populations[-1] / self.populations[0]) ** (1 / (len(self) - 1)) - 1


class Forecaster(ABC):
    """Shared interface for one-step or multi-year forecasts."""

    # ABC makes subclasses implement both methods before they can be used.

    @abstractmethod
    def fit(self, values):
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        pass


class LinearTrend(Forecaster):
    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        if len(self.values) < 2:
            raise ValueError('Linear trend requires at least two observations')
        # polyfit finds the straight line that minimizes squared fitting errors.
        self.slope, self.intercept = np.polyfit(np.arange(len(self.values)), self.values, 1)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError('Horizon must be positive')
        x = np.arange(len(self.values), len(self.values) + horizon)
        return np.maximum(0, self.slope * x + self.intercept)

    def fitted(self) -> np.ndarray:
        return np.maximum(0, self.slope * np.arange(len(self.values)) + self.intercept)


class ExponentialGrowth(Forecaster):
    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        if len(self.values) < 2 or self.values[0] <= 0 or self.values[-1] <= 0:
            raise ValueError('CAGR requires positive endpoints and at least two values')
        # Use the first and last observations to estimate one constant annual rate.
        self.rate = (self.values[-1] / self.values[0]) ** (1 / (len(self.values)-1)) - 1
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError('Horizon must be positive')
        return self.values[-1] * (1 + self.rate) ** np.arange(1, horizon + 1)


class FibonacciRatio(Forecaster):
    """Illustrative recurrence; not a demographic model."""

    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        if len(self.values) < 2:
            raise ValueError('At least two observations are required')
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if horizon < 1:
            raise ValueError('Horizon must be positive')
        a, b = 1, 1
        last = float(self.values[-1])
        forecasts = []
        for _ in range(horizon):
            # Consecutive Fibonacci values supply the next multiplier.
            # This is for comparison, not a claim about real population growth.
            a, b = b, a + b
            last *= b / a
            forecasts.append(last)
        return np.array(forecasts)


def error_metrics(actual, predicted) -> dict:
    """Compare strictly positive actual values (MAPE needs nonzero denominators)."""
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) == 0 or a.shape != p.shape or np.any(a <= 0):
        raise ValueError('Provide matching arrays with positive actual values')
    err = a - p
    # MAPE divides by actual population; zero actuals would make it undefined.
    return {'MAE': float(np.mean(abs(err))), 'RMSE': float(np.sqrt(np.mean(err**2))),
            'MAPE (%)': float(np.mean(abs(err / a)) * 100)}


def bootstrap_interval(model, train, horizon: int, seed=42, repeats=1200):
    """Resample past one-step errors; a conditional, approximate interval."""
    train = np.asarray(train, dtype=float)
    if len(train) < 5:
        raise ValueError('Five observations are needed for residual resampling')
    # For each past year, predict it using only the earlier years. The errors
    # become examples of how far future forecasts might miss the true value.
    residuals = np.array([train[i]-type(model)().fit(train[:i]).predict(1)[0]
                          for i in range(3, len(train))])
    rng = np.random.default_rng(seed)
    # Resample those errors; 2.5% and 97.5% delimit an approximate interval.
    draws = model.predict(horizon)[None, :] + rng.choice(residuals, (repeats, horizon))
    return np.maximum(0, np.quantile(draws, [0.025, 0.975], axis=0))
