import numpy as np
import scipy.special as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import cache


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


def transform(func, R, M, K, r_dim=200, theta_dim=200):
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
    r_dim : int, optional
        Number of radial sample points.
    theta_dim : int, optional
        Number of angular sample points.

    Returns
    -------
    coeffs : np.ndarray
        A 2D array of shape (2*M+1, K) containing the spectral coefficients A_mk.
    """
    alphas = get_alphas(M, K)  # (2M+1, K)

    Rs, Ts = np.ogrid[0 : R : r_dim * 1j, 0 : 2 * np.pi : 2 * np.pi / theta_dim]  # (r_dim, theta_dim)

    dr = R / (r_dim - 1)
    dt = (2 * np.pi) / theta_dim

    Frdrdt = func(Rs, Ts) * Rs * dr * dt  # (r_dim, theta_dim)

    m = np.arange(-M, M + 1)  # (2M+1,)
    r = Rs.ravel()  # (r,)

    E = np.exp(1j * m[:, None] * Ts)  # (2M+1, theta_dim)
    J = sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :] / R)  # (2M+1, K, r_dim)

    coeffs = np.einsum("rt,mt,mkr->mk", Frdrdt, E, J, optimize=True)  # (2M+1, K)
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
    M_full, K = coeffs.shape
    M = (M_full - 1) // 2
    alphas = get_alphas(M, K)  # (2M+1, K)

    Ys, Xs = np.mgrid[-R : R : N * 1j, -R : R : N * 1j]  # (N, N)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N, N)

    m = np.arange(-M, M + 1)

    J = sp.jv(m[:, None, None, None], alphas[:, :, None, None] * (Rs[None, None, :, :] / R))  # (2M+1, K, N, N)
    E = np.exp(-1j * m[:, None, None] * Ts[None, :, :])  # (2M+1, N, N)

    f = np.einsum("mk,mkxy,mxy->xy", coeffs, J, E, optimize=True)  # (N, N)
    f = np.real(f)
    f[Rs > R] = np.nan

    return Xs, Ys, f


if __name__ == "__main__":

    def example_func(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z1 = np.exp(-((x - 80) ** 2 + y**2) / (2 * 40**2))
        z2 = -0.8 * np.exp(-((x + 80) ** 2 + (y - 50) ** 2) / (2 * 30**2))
        return z1 + z2

    RADIUS = 256
    M_MAX = 10  # angular orders
    K_MAX = 5  # radial modes

    # Transform
    coeffs = transform(example_func, R=RADIUS, M=M_MAX, K=K_MAX)

    # Reconstruct
    X, Y, Z_rec = inverse_transform(coeffs, R=RADIUS, N=200)

    # Get ground truth
    R_grid = np.sqrt(X**2 + Y**2)
    T_grid = np.arctan2(Y, X)
    Z_true = example_func(R_grid, T_grid)
    Z_true[R_grid > RADIUS] = np.nan

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
