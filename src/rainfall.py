"""Monthly rainfall comparison and simple crop moisture screening."""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as scipy_cosine_distance

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


class Region:
    """Twelve illustrative monthly rainfall totals in millimetres."""

    def __init__(self, name: str, rainfall):
        self.name = name
        self.rainfall = np.asarray(rainfall, dtype=float)
        if self.rainfall.shape != (12,) or np.any(~np.isfinite(self.rainfall)) or np.any(self.rainfall < 0):
            raise ValueError('Provide 12 finite, non-negative monthly values')

    def annual_total(self) -> float:
        return float(np.sum(self.rainfall))

    def mean(self) -> float:
        return float(np.mean(self.rainfall))

    def wettest(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        if self.mean() == 0:
            raise ValueError('CV is undefined when rainfall is zero throughout')
        return float(np.std(self.rainfall, ddof=1)/self.mean())

    def rainy_peaks(self, prominence=25) -> list:
        """Find monthly peaks with Dec and Jan treated as adjacent."""
        extended = np.r_[self.rainfall[-2:], self.rainfall, self.rainfall[:2]]
        indexes, _ = find_peaks(extended, prominence=prominence, distance=2)
        return sorted({int(i-2) for i in indexes if 2 <= i < 14})


class CropRule:
    """Illustrative MONTHLY moisture screen, not a yield or planting model."""

    def __init__(self, crop: str, minimum: float, maximum: float):
        if minimum < 0 or maximum <= minimum:
            raise ValueError('Rainfall band must satisfy 0 <= minimum < maximum')
        self.crop, self.minimum, self.maximum = crop, minimum, maximum

    def classify(self, mm: float) -> str:
        if mm < 0:
            raise ValueError('Rainfall cannot be negative')
        return 'Drought risk' if mm < self.minimum else ('Waterlogging risk' if mm > self.maximum else 'Suitable')


def cosine_similarity(a, b) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or np.linalg.norm(x)*np.linalg.norm(y) == 0:
        raise ValueError('Equal-length nonzero vectors are required')
    return float(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)))


def compare(a, b) -> dict:
    x, y = a.rainfall, b.rainfall
    if np.std(x) == 0 or np.std(y) == 0:
        correlation = float('nan')
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    return {'cosine': cosine_similarity(x,y), 'pearson': correlation,
            'euclidean_mm': float(np.linalg.norm(x-y)),
            'scipy_cosine': float(1-scipy_cosine_distance(x,y))}
