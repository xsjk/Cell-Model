from functools import cache

import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.special as sp
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator, interp1d


@cache
def get_alphas(M: int, K: int) -> np.ndarray:
    """
    Get the first K zeros of the Bessel function of order m.

    Parameters
    ----------
    M : int
        The maximum order of the Bessel function.
    K : int
        The number of zeros to compute for each order.

    Returns
    -------
    alphas : np.ndarray
        A 2D array of shape (2*M+1, K) containing the Bessel function zeros alpha_mk.
    """
    m = np.arange(-M, M + 1)  # (2M+1,)
    jnz = np.array([sp.jn_zeros(n, K) for n in range(M + 1)])  # (M+1, K)
    return jnz[np.abs(m)]  # (2M+1, K)


@cache
def get_bessel_basis(M: int, K: int, N_r: int) -> np.ndarray:
    """
    Precompute Bessel basis functions J_m(alpha_mk * r) on a normalized radial grid [0, 1].

    Parameters
    ----------
    M : int
        Maximum angular order.
    K : int
        Number of radial zeros.
    N_r : int
        Number of radial points.

    Returns
    -------
    J : np.ndarray
        Shape (2*M+1, K, N_r).
    """
    alphas = get_alphas(M, K)  # (2M+1, K)
    m = np.arange(-M, M + 1)  # (2M+1,)
    r = np.linspace(0, 1, N_r)  # (N_r,)

    return sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :])  # (2M+1, K, N_r)


def transform(func, R, M, K, N_r=200, N_theta=200):
    """
    Compute the Fourier-Bessel spectral coefficients of a function defined on a disk.
    Fully vectorized implementation using einsum and broadcasting.

    Parameters
    ----------
    func : callable
        A function of two variables (r, theta) defined on the disk of radius R.
    R : float
        The radius of the disk.
    M : int
        The maximum angular order (m) for the decomposition.
    K : int
        The number of radial modes (k) for each angular order.
    N_r : int, optional
        Number of radial sample points.
    N_theta : int, optional
        Number of angular sample points.

    Returns
    -------
    coeffs : np.ndarray
        A 2D array of shape (2*M+1, K) containing the spectral coefficients A_mk.
    """
    alphas = get_alphas(M, K)  # (2M+1, K)

    Rs, Ts = np.ogrid[0 : R : N_r * 1j, 0 : 2 * np.pi : 2 * np.pi / N_theta]  # (N_r, 1), (1, N_theta)

    dr, dt = R / (N_r - 1), 2 * np.pi / N_theta
    Frdrdt = func(Rs, Ts) * Rs * dr * dt  # (N_r, N_theta)

    m = np.arange(-M, M + 1)  # (2M+1,)

    E = np.exp(-1j * m[:, None] * Ts)  # (2M+1, N_theta)
    J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)

    coeffs = np.einsum("rt, mt, mkr -> mk", Frdrdt, E, J, optimize=True)  # (2M+1, K)
    coeffs /= np.pi * (R**2) * (sp.jv(m[:, None] + 1, alphas) ** 2)  # (2M+1, K)
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


def transform_fast(func, R, M, K, N_r=200, N_theta=200):
    Rs = np.linspace(0, R, N_r)  # (N_r,)
    Ts = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    m = np.arange(-M, M + 1)  # (2M+1,)

    f = func(Rs[:, None], Ts[None, :])  # (N_r, N_theta)

    # FFT along Theta (axis 1): N_theta -> 2M+1
    dt = 2 * np.pi / N_theta
    dr = R / (N_r - 1)
    F: np.ndarray = scipy.fft.fft(f, axis=1)  # type: ignore
    F = F[:, m]  # (N_r, 2M+1)

    # Radial projection along R (axis 0)
    J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    alphas = get_alphas(M, K)  # (2M+1, K)
    coeffs = np.einsum("rm, r, mkr -> mk", F * dt, Rs * dr, J)  # (2M+1, K)
    coeffs /= np.pi * (R**2) * (sp.jv(m[:, None] + 1, alphas) ** 2)
    return coeffs


def inverse_transform_fast(coeffs, R, N=256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, K = (coeffs.shape[0] - 1) // 2, coeffs.shape[1]
    N_r = int(np.hypot(N, N))
    N_theta = int(2 * np.pi * N)

    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    r = np.linspace(0, R, N_r)  # (N_r,)

    # Radial Summation along R (axis 1)
    J = get_bessel_basis(M, K, N_r)  # (2M+1, K, N_r)
    C = np.einsum("mk, mkr -> mr", coeffs, J)  # (2M+1, N_r)

    # IFFT along Theta (axis 0): 2M+1 -> N_theta
    F = np.zeros((N_theta, N_r), dtype=np.complex128)  # (N_theta, N_r)
    F[m] = C

    f: np.ndarray = scipy.fft.ifft(F, axis=0)  # type: ignore
    f = np.real(f) * N_theta  # (N_theta, N_r)
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
    M_MAX = 5  # angular orders
    K_MAX = 5  # radial modes
    N_R = 500
    N_THETA = 50

    # Transform
    coeffs = transform(example_func, R=RADIUS, M=M_MAX, K=K_MAX, N_r=N_R, N_theta=N_THETA)
    coeffs_fast = transform_fast(example_func, R=RADIUS, M=M_MAX, K=K_MAX, N_r=N_R, N_theta=N_THETA)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    X, Y, Z_rec = inverse_transform(coeffs, R=RADIUS, N=N_R)
    X_f, Y_f, Z_rec_fast = inverse_transform_fast(coeffs, R=RADIUS, N=N_R)
    print(f"Inverse Transform max diff: {np.max(np.abs(Z_rec - Z_rec_fast)):.6e}")

    # Get ground truth
    R_grid = np.sqrt(X**2 + Y**2)
    T_grid = np.arctan2(Y, X)
    Z_true = example_func(R_grid, T_grid)
    Z_true[R_grid > RADIUS] = np.nan
    Z_rec[R_grid > RADIUS] = np.nan

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
