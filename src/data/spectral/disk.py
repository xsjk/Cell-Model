from functools import cache
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.special as sp
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import roots_legendre


@cache
def get_alphas(M: int, K: int) -> np.ndarray:
    """
    Get the first K zeros of the Bessel function of order m.
    """
    m = np.arange(-M, M + 1)  # (2M+1,)
    jnz = np.array([sp.jn_zeros(n, K) for n in range(M + 1)])  # (M+1, K)
    return jnz[np.abs(m)]  # (2M+1, K)


@cache
def get_bessel_basis(M: int, K: int, N_r: int) -> np.ndarray:
    """
    Precompute Bessel basis functions J_m(alpha_mk * r) on a normalized linear radial grid [0, 1].
    Used for interpolation-based reconstruction or Simpson integration.
    """
    alphas = get_alphas(M, K)  # (2M+1, K)
    m = np.arange(-M, M + 1)  # (2M+1,)
    r = np.linspace(0, 1, N_r)  # (N_r,)

    return sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :])  # (2M+1, K, N_r)


@cache
def get_legendre_basis(M: int, K: int, N_r: int):
    """
    Precompute Gauss-Legendre nodes, weights and Bessel basis functions.
    Used for high-precision spectral integration.
    """
    x, w = roots_legendre(N_r)  # (N_r,), (N_r,)
    # Map [-1, 1] to [0, 1] for normalized radial coordinate
    r_norm = (x + 1) / 2  # (N_r,)
    w_norm = w / 2  # (N_r,)

    alphas = get_alphas(M, K)  # (2M+1, K)
    m = np.arange(-M, M + 1)  # (2M+1,)

    J = sp.jv(m[:, None, None], alphas[:, :, None] * r_norm[None, None, :])  # (2M+1, K, N_r)
    return r_norm, w_norm, J


@cache
def get_bessel_norm_factors(M: int, K: int) -> np.ndarray:
    """
    Get normalization factors for Fourier-Bessel coefficients.
    """
    m = np.arange(-M, M + 1)  # (2M+1,)
    alphas = get_alphas(M, K)  # (2M+1, K)
    return np.pi * sp.jv(m[:, None] + 1, alphas) ** 2  # (2M+1, K)


def transform(func, R, M, K, N_r=200, N_theta=200, method: Literal["legendre", "simpson"] = "legendre"):
    """
    Compute the Fourier-Bessel spectral coefficients of a function defined on a disk.
    Uses direct angular integration instead of FFT.

    Parameters
    ----------
    func : callable
        A function of two variables (r, theta).
    R : float
        Radius of disk.
    method : str
        'legendre': Uses Gauss-Legendre quadrature (Spectral accuracy, recommended).
                    N_r corresponds to number of quadrature nodes.
        'simpson': Uses Simpson's rule on uniform grid (Algebraic accuracy).
    """
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    dt = 2 * np.pi / N_theta

    if method == "legendre":
        r_norm, w_norm, J = get_legendre_basis(M, K, N_r)
        r = R * r_norm  # (N_r,)
        Ws = R * w_norm  # (N_r,)
    elif method == "simpson":
        r = np.linspace(0, R, N_r)  # (N_r,)
        J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    else:
        raise ValueError(f"Unknown method: {method}")

    f = func(r[:, None], t[None, :])  # (N_r, N_theta)

    E = np.exp(-1j * m[:, None] * t[None, :])  # (2M+1, N_theta)
    F = np.einsum("rt, mt -> rm", f, E, optimize=True) * dt  # (N_r, 2M+1)

    if method == "legendre":
        coeffs = np.einsum("rm, r, mkr -> mk", F, r * Ws, J, optimize=True)  # (2M+1, K)
    else:  # simpson
        coeffs = simpson(np.einsum("rm, r, mkr -> rmk", F, r, J, optimize=True), x=r, axis=0)  # (2M+1, K)
    coeffs /= get_bessel_norm_factors(M, K) * R**2  # (2M+1, K)
    return coeffs


def transform_fast(func, R, M, K, N_r=200, N_theta=200, method: Literal["legendre", "simpson"] = "legendre"):
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    dt = 2 * np.pi / N_theta

    if method == "legendre":
        r_norm, w_norm, J = get_legendre_basis(M, K, N_r)
        r = R * r_norm  # (N_r,)
        Ws = R * w_norm  # (N_r,)
    elif method == "simpson":
        r = np.linspace(0, R, N_r)  # (N_r,)
        J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    else:
        raise ValueError(f"Unknown method: {method}")

    f = func(r[:, None], t[None, :])  # (N_r, N_theta)

    # FFT along Theta (axis 1)
    F: np.ndarray = scipy.fft.fft(f, axis=1) * dt  # (N_r, N_theta) # type: ignore
    F = F[:, m]  # (N_r, 2M+1)

    # DHT along R (axis 0): N_r -> K
    if method == "legendre":
        coeffs = np.einsum("rm, r, mkr -> mk", F, r * Ws, J, optimize=True)  # (2M+1, K)
    else:  # simpson
        coeffs = simpson(np.einsum("rm, r, mkr -> rmk", F, r, J, optimize=True), x=r, axis=0)  # (2M+1, K)
    coeffs /= get_bessel_norm_factors(M, K) * R**2  # (2M+1, K)
    return coeffs


def inverse_transform(coeffs, R, N=256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the function from its Fourier-Bessel spectral coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        2D array of shape (2*M+1, K) containing the spectral coefficients A_mk.
    R : float
        The radius of the disk.
    N : int, optional
        The size of the output grid (N x N).

    Returns
    -------
    Xs : np.ndarray
        The x-coordinates of the grid.
    Ys : np.ndarray
        The y-coordinates of the grid.
    f : np.ndarray
        The reconstructed function values on the grid.
    """
    M, K = (coeffs.shape[0] - 1) // 2, coeffs.shape[1]
    N_r = int(np.hypot(N, N))

    Ys, Xs = np.mgrid[-R : R : N * 1j, -R : R : N * 1j]  # (N, N)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N, N)

    r = np.linspace(0, R, N_r)  # (N_r,)
    m = np.arange(-M, M + 1)  # (2M+1,)

    J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    E = np.exp(1j * m[:, None, None] * Ts[None, :, :])  # (2M+1, N, N)

    C = np.einsum("mk, mkr -> mr", coeffs, J)  # (2M+1, N_r)
    C = interp1d(r, C, axis=-1, bounds_error=False, fill_value=0)(Rs)  # (2M+1, N, N)
    f = np.einsum("mxy, mxy -> xy", C, E, optimize=True)  # (N, N)

    return Xs, Ys, np.real(f)


def inverse_transform_fast(coeffs, R, N=256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, K = (coeffs.shape[0] - 1) // 2, coeffs.shape[1]
    N_r = int(np.hypot(N, N))
    N_theta = int(2 * np.pi * N)

    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    r = np.linspace(0, R, N_r)  # (N_r,)

    # IDHT along R (axis 1): K -> N_r
    J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    C = np.einsum("mk, mkr -> mr", coeffs, J)  # (2M+1, N_r)

    # IFFT along Theta (axis 0): 2M+1 -> N_theta
    F = np.zeros((N_theta, N_r), dtype=np.complex128)  # (N_theta, N_r)
    F[m] = C
    f: np.ndarray = scipy.fft.ifft(F, axis=0).real * N_theta  # (N_theta, N_r) # type: ignore
    f = np.pad(f, ((0, 1), (0, 0)), mode="wrap")  # (N_theta+1, N_r)

    # Interpolate Polar -> Cartesian
    Ys, Xs = np.mgrid[-R : R : N * 1j, -R : R : N * 1j]  # (N, N)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N, N)
    grid = np.stack([np.mod(Ts, 2 * np.pi), Rs], axis=-1)  # (N, N, 2)
    f = RegularGridInterpolator((t, r), f, bounds_error=False, fill_value=0)(grid)  # (N, N)

    return Xs, Ys, f


if __name__ == "__main__":

    def example_func(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z1 = np.exp(-((x - 80) ** 2 + y**2) / (2 * 40**2))
        z2 = -0.8 * np.exp(-((x + 80) ** 2 + (y - 50) ** 2) / (2 * 30**2))
        return z1 + z2

    RADIUS = 256
    M_MAX = 20  # angular orders
    K_MAX = 20  # radial modes
    N_R = 500
    N_THETA = 50

    # Transform (Normal vs Fast)
    coeffs_normal = transform(example_func, R=RADIUS, M=M_MAX, K=K_MAX, N_r=N_R, N_theta=N_THETA, method="legendre")
    coeffs_fast = transform_fast(example_func, R=RADIUS, M=M_MAX, K=K_MAX, N_r=N_R, N_theta=N_THETA, method="legendre")
    print(f"Transform Diff (Normal vs Fast): {np.max(np.abs(coeffs_normal - coeffs_fast)):.6e}")

    # Reconstruct (Inverse vs Inverse Fast)
    X, Y, Z_rec = inverse_transform(coeffs_normal, R=RADIUS, N=N_R)
    _, _, Z_rec_fast = inverse_transform_fast(coeffs_normal, R=RADIUS, N=N_R)
    print(f"Reconstruction Diff (Regular vs Fast): {np.nanmax(np.abs(Z_rec - Z_rec_fast)):.6e}")

    # Get ground truth
    R_grid = np.sqrt(X**2 + Y**2)
    T_grid = np.arctan2(Y, X)
    Z_true = example_func(R_grid, T_grid)
    Z_true[R_grid > RADIUS] = np.nan
    Z_rec[R_grid > RADIUS] = np.nan

    # Display errors
    print("\nError vs Ground Truth:")
    print(f"MAE: {np.nanmax(np.abs(Z_true - Z_rec)):.6e}")
    print(f"RMSE: {np.sqrt(np.nanmean((Z_true - Z_rec) ** 2)):.6e}")

    # Visualization
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("Original Function f(r, theta)", f"Reconstructed (M={M_MAX}, K={K_MAX})"),
    )
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_true, colorscale="Viridis", showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_rec, colorscale="Viridis", showscale=False), row=1, col=2)
    fig.update_layout(
        title_text="Fourier-Bessel Spectral Decomposition on Disk",
        template="plotly_dark",
        height=800,
        scene=dict(aspectratio=dict(x=1, y=1, z=0.5)),
        scene2=dict(aspectratio=dict(x=1, y=1, z=0.5)),
    )
    fig.write_html("fourier_bessel_decomposition.html")
