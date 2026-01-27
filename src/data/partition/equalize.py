from typing import Callable

import numpy as np


def compute_cdf(data: np.ndarray, bins: int) -> Callable[[np.ndarray], np.ndarray]:
    assert data.ndim == 1
    assert data.size > 0
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    cdf = np.insert(cdf, 0, 0)
    return lambda x: np.interp(x, bin_edges, cdf)


def equalize_scalar_field(s: np.ndarray, mask=None, bins: int = 1000000) -> np.ndarray:
    """
    Equalize histogram of a scalar field s in mask region.
    Map s to uniform distribution [0,1].

    Parameters
    ----------
    s : np.ndarray
        Input scalar field.
    mask : np.ndarray, optional
        Boolean mask defining the region to equalize. If None, equalize over entire field.
    bins : int
        Number of bins for histogram to approximate CDF.

    Returns
    -------
    s_eq : np.ndarray
        Equalized scalar field.
    """
    if mask is None:
        mask = ...

    mapper = compute_cdf(s[mask], bins=bins)
    s_eq = s.copy()
    s_eq[mask] = mapper(s[mask])

    return s_eq
