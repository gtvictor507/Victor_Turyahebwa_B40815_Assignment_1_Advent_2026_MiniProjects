"""Route revenue, demand and straightforward passenger forecasters."""

import math
import statistics
from abc import ABC, abstractmethod

import numpy as np
from scipy import linalg


class Route:
    """One route's daily passenger totals and fare per passenger (UGX)."""

    def __init__(self, name: str, counts, fare: float):
        self.name, self.counts, self.fare = name, np.asarray(counts, dtype=float), float(fare)
        if self.counts.ndim != 1 or len(self.counts) < 4 or np.any(~np.isfinite(self.counts)) or np.any(self.counts < 0):
            raise ValueError('Provide at least four finite non-negative passenger counts')
        if not math.isfinite(self.fare) or self.fare < 0:
            raise ValueError('Fare must be finite and non-negative')

    def daily_revenue(self) -> np.ndarray:
        return self.counts*self.fare

    def total_revenue(self) -> float:
        return float(np.sum(self.daily_revenue()))

    def statistics(self) -> dict:
        values = self.counts.tolist()
        return {'mean': statistics.mean(values), 'variance': statistics.variance(values),
                'std_dev': statistics.stdev(values)}


class Forecaster(ABC):
    @abstractmethod
    def fit(self, values):
        pass

    @abstractmethod
    def predict(self) -> float:
        pass


class MovingAverage(Forecaster):
    def fit(self, values):
        self.values = np.asarray(values, dtype=float)
        if len(self.values) < 3:
            raise ValueError('Need three days of data')
        return self

    def predict(self) -> float:
        return float(np.mean(self.values[-3:]))


class ExponentialSmoothing(Forecaster):
    def __init__(self, alpha=0.5):
        if not 0 < alpha <= 1:
            raise ValueError('Alpha must be between 0 (exclusive) and 1')
        self.alpha = alpha

    def fit(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            raise ValueError('Need two days of data')
        self.level = float(values[0])
        for value in values[1:]:
            self.level = self.alpha*value+(1-self.alpha)*self.level
        return self

    def predict(self) -> float:
        return self.level


class LinearTrend(Forecaster):
    def fit(self, values):
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            raise ValueError('Need two days of data')
        self.slope, self.intercept = np.polyfit(np.arange(len(values)), values, 1)
        self.next_day = len(values)
        return self

    def predict(self) -> float:
        return max(0., float(self.slope*self.next_day+self.intercept))


def equilibrium() -> tuple:
    """Solve 120 - 0.02P = Q and 10 + 0.03P = Q."""
    fare, passengers_per_trip_hour = linalg.solve(
        np.array([[.02, 1], [-.03, 1]]), np.array([120, 10]))
    return float(fare), float(passengers_per_trip_hour)


def walk_forward(values, factory, first_day=4) -> tuple:
    """Forecast days 4-10 using only earlier days (one-based day labels)."""
    data = np.asarray(values, dtype=float)
    predictions = np.array([factory().fit(data[:day-1]).predict()
                            for day in range(first_day, len(data)+1)])
    actual = data[first_day-1:]
    return float(np.mean(abs(predictions-actual))), predictions


def vehicles_needed(passengers: float, seats=14, trips=8, buffer=.15) -> int:
    if passengers < 0 or seats <= 0 or trips <= 0 or buffer < 0:
        raise ValueError('Demand and fleet inputs must be non-negative')
    return math.ceil(passengers*(1+buffer)/(seats*trips))
