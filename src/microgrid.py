"""Two-source micro-grid balance with non-negative fallback."""

import csv
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.optimize import nnls


class MicroGrid:
    """Solve two demand equations for solar and battery energy in kWh."""

    def __init__(self):
        # Rows represent demand equations; columns represent solar and battery.
        self.matrix = np.array([[3.0, 2.0], [4.0, 1.0]])

    def diagnostics(self) -> tuple:
        # A nonzero determinant means a unique algebraic solution. A large
        # condition number means demand errors could strongly affect it.
        return float(np.linalg.det(self.matrix)), float(np.linalg.cond(self.matrix))

    def solve_day(self, d1: float, d2: float) -> dict:
        values = np.array([d1, d2], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Demand values must be finite and non-negative')
        if abs(np.linalg.det(self.matrix)) < 1e-10 or np.linalg.cond(self.matrix) > 1e10:
            raise ValueError('Demand system is singular or ill-conditioned')
        raw = linalg.solve(self.matrix, values)
        if np.all(raw >= -1e-9):
            feasible = np.maximum(raw, 0)
            return {'solar': feasible[0], 'battery': feasible[1], 'infeasible': False,
                    'residual': float(np.linalg.norm(self.matrix @ feasible-values))}
        # Exact equations sometimes require negative energy. NNLS finds the
        # closest non-negative assignment; residual shows the unmatched demand.
        feasible, residual = nnls(self.matrix, values)
        return {'solar': feasible[0], 'battery': feasible[1], 'infeasible': True,
                'residual': float(residual)}

    def solve_many(self, demands: np.ndarray) -> np.ndarray:
        """One vectorised 2 x n solve; raw solution before nnls corrections."""
        data = np.asarray(demands, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2 or np.any(~np.isfinite(data)) or np.any(data < 0):
            raise ValueError('Demands must be a non-negative, finite n x 2 array')
        # Transpose from n rows of days to a 2 x n right-hand side for SciPy.
        return linalg.solve(self.matrix, data.T).T


def get_demand(input_func=input) -> float:
    """Prompt until a valid demand is provided; injectable input helps testing."""
    while True:
        try:
            # Reject blanks, text, negative values and infinity; then re-prompt.
            number = float(input_func('Enter demand in kWh: ').strip())
            if math.isfinite(number) and number >= 0:
                return number
        except ValueError:
            pass
        print('Please enter a non-negative finite number.')


def generate_demands(path: str, seed=42) -> None:
    """Generate 30 reproducible illustrative days, including two infeasible ones."""
    rng = np.random.default_rng(seed)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['day', 'D1_kWh', 'D2_kWh'])
        for day in range(1, 31):
            # The weekend shift produces a repeated weekly demand pattern.
            weekend = (day-1) % 7 in (5, 6)
            solar = max(1, rng.normal(25 if not weekend else 21, 2))
            battery = max(1, rng.normal(17 if not weekend else 21, 2))
            d1, d2 = 3*solar+2*battery, 4*solar+battery
            d1 += rng.normal(0, 0.5)
            d2 += rng.normal(0, 0.5)
            if day in (10, 24):
                # Include two cases where the exact equations need negative y.
                d2 = 1.5*d1  # violates the non-negative battery constraint
            writer.writerow([day, round(d1, 3), round(d2, 3)])


def read_demands(path: str) -> np.ndarray:
    with Path(path).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('CSV contains no demand rows')
    try:
        result = np.array([[float(row['D1_kWh']), float(row['D2_kWh'])] for row in rows])
    except (KeyError, ValueError) as exc:
        raise ValueError('CSV needs numeric D1_kWh and D2_kWh columns') from exc
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError('Demands must be finite and non-negative')
    return result


def describe(values) -> dict:
    data = list(map(float, values))
    return {'mean': statistics.mean(data), 'variance': statistics.variance(data),
            'std_dev': statistics.stdev(data)}
