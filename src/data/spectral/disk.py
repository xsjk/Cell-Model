from functools import cache
from typing import Literal

import numpy as np
import scipy.fft
import scipy.special as sp
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d


def soft_mask(r, R=1, margin=None):
    """
    Create a soft mask that smoothly transitions from 1 to 0 near the boundary of the disk of radius R.
    Use cosine tapering over the specified margin.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinates.
    R : float
        Radius of the disk.
    margin : float, optional
        Width over which to apply the smooth transition.
    """
    if margin is None:
        margin = R
    assert 0 <= margin <= R, "margin must be less than R"
    if margin == 0:
        return (r <= R).astype(float)
    y = 0.5 * (1 + np.cos(np.pi * (r - (R - margin)) / margin))
    return (r < R - margin) + (r >= R - margin) * y


@cache
def _get_alphas(M: int, K: int) -> np.ndarray:
    return np.array([sp.jn_zeros(n, K) for n in range(M + 1)])  # (M+1, K)


@cache
def bessel_basis(M: int, K: int, N_r: int) -> tuple[np.ndarray, np.ndarray]:
    alphas = _get_alphas(M, K)  # (M+1, K)
    m = np.arange(0, M + 1)  # (M+1,)
    r = np.linspace(0, 1, N_r)  # (N_r,)

    return r, sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :])  # (M+1, K, N_r)


@cache
def legendre_basis(M: int, K: int, N_r: int):
    x, w = sp.roots_legendre(N_r)  # (N_r,), (N_r,)
    # Map [-1, 1] to [0, 1] for normalized radial coordinate
    r = (x + 1) / 2  # (N_r,)
    w = w / 2  # (N_r,)

    alphas = _get_alphas(M, K)  # (M+1, K)
    m = np.arange(0, M + 1)  # (M+1,)

    J = sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :])  # (M+1, K, N_r)
    return r, w, J


@cache
def norm_factors(M: int, K: int) -> np.ndarray:
    m = np.arange(0, M + 1)  # (M+1,)
    alphas = _get_alphas(M, K)  # (M+1, K)
    return np.pi * sp.jv(m[:, None] + 1, alphas) ** 2  # (M+1, K)


@cache
def dht_basis(M: int, K: int):
    m = np.arange(0, M + 1)  # (M+1, 1)
    alphas = _get_alphas(M, K + 1)  # (M+1, K+1)
    r = alphas[:, :K] / alphas[:, [K]]  # (M+1, K)
    J = sp.jv(m[:, None, None], alphas[:, :K, None] * r[:, None, :])  # (M+1, K, K)
    w = 2.0 / (alphas[:, [K]] ** 2 * norm_factors(M, K) / np.pi)  # (M+1, K)
    return r, w, J


def min_n_theta(M) -> int:
    """
    Minimum number of angular sample points to avoid aliasing for max angular order M.
    """
    return scipy.fft.next_fast_len(int(2 * M + 1), real=True)  # type: ignore


def transform(func, R, M, K, N_r=200, N_theta=None, method: Literal["legendre", "simpson", "dht"] = "legendre"):
    """
    Compute the Fourier-Bessel spectral coefficients of a real-valued function defined on a disk.

        **Domain**:
            Disk = B^2 = {(r, theta) | 0 <= r <= R, 0 <= theta < 2pi}

        **Basis**:
            Fourier-Bessel basis J_m(alpha_mk * r / R) * exp(i * m * theta)
            for m = 0, 1, ..., M and k = 1, 2, ..., K,
            where J_m is the Bessel function of the first kind of order m,
            and alpha_mk is the k-th positive zero of J_m.

    Parameters
    ----------
    func : callable
        Function f(r, theta).
    R : float
        Radius.
    M : int
        Max angular order (0 <= m <= M).
    K : int
        Number of radial modes.
    N_r : int
        Number of radial sample points.
    N_theta : int
        Number of angular sample points.
    method : str
        'legendre': Uses Gauss-Legendre quadrature for radial direction.
        'simpson': Uses Simpson's rule on uniform radial grid.
        'dht': Uses Discrete Hankel Transform.

    Returns
    -------
    coeffs : np.ndarray
        Shape (M+1, K).

    Notes
    -----
    For real-valued functions, only m >= 0 coefficients are returned,
    since the m < 0 coefficients can be obtained via conjugation.
    """
    N_theta = N_theta or min_n_theta(M)

    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    m = np.arange(0, M + 1)  # (M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    dt = 2 * np.pi / N_theta

    E = np.exp(-1j * m[:, None] * t[None, :])  # (M+1, N_theta)

    match method:
        case "legendre":
            r, w, J = legendre_basis(M, K, N_r)  # (N_r,), (N_r,), (M+1, K, N_r)
            f = func(r[:, None] * R, t[None, :])  # (N_r, N_theta)
            coeffs = np.einsum("rt, mt, r, r, mkr -> mk", f, E, r, w, J, optimize=True) * dt  # (M+1, K)
        case "simpson":
            r, J = bessel_basis(M, K, N_r)  # (N_r,), (M+1, K, N_r)
            f = func(r[:, None] * R, t[None, :])  # (N_r, N_theta)
            coeffs = simpson(np.einsum("rt, mt, r, mkr -> rmk", f, E, r, J, optimize=True), x=r, axis=0) * dt  # (M+1, K)
        case "dht":
            r, w, J = dht_basis(M, K)  # (M+1, K), (M+1, K), (M+1, K, K)
            f = func(r[:, :, None] * R, t[None, None, :])  # (M+1, K, N_theta)
            coeffs = np.einsum("mrt, mt, mkr, mr -> mk", f, E, J, w, optimize=True) * dt  # (M+1, K)
        case _:
            raise ValueError(f"Unknown method: {method}")

    coeffs /= norm_factors(M, K)  # (M+1,K)
    return coeffs


def inverse_transform(coeffs, R, N=256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the real-valued function on a Cartesian grid from its Fourier-Bessel coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Fourier-Bessel coefficients of shape (M+1, K).
    R : float
        The radius of the disk.
    N : int, optional
        The size of the output grid (N x N).

    Returns
    -------
    Xs, Ys : np.ndarray
        The grid coordinates.
    f : np.ndarray
        The reconstructed function values on the grid.
    """
    M, K = coeffs.shape[0] - 1, coeffs.shape[1]
    N_r = int(np.hypot(N, N))

    Ys, Xs = np.mgrid[-R : R : N * 1j, -R : R : N * 1j]  # (N, N)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N, N)

    m = np.arange(0, M + 1)  # (M+1,)

    r, J = bessel_basis(M, K, N_r)  # (N_r,), (M+1, K, N_r)
    E = np.exp(1j * m[:, None, None] * Ts[None, :, :])  # (M+1, N, N)

    C = np.einsum("mk, mkr -> mr", coeffs, J)  # (M+1, N_r)
    C = interp1d(r * R, C, axis=-1, bounds_error=False, fill_value=0)(Rs)  # (M+1, N, N)
    f = C[0].real * E[0].real + 2 * np.einsum("mxy, mxy -> xy", C[1:], E[1:], optimize=True).real  # (N, N)
    return Xs, Ys, f


def transform_fast(func, R, M, K, N_r=200, N_theta=None, method: Literal["legendre", "simpson"] = "legendre"):
    N_theta = N_theta or min_n_theta(M)
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    dt = 2 * np.pi / N_theta
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)

    match method:
        case "legendre":
            r, w, J = legendre_basis(M, K, N_r)  # (N_r,), (N_r,), (M+1, K, N_r)
        case "simpson":
            r, J = bessel_basis(M, K, N_r)  # (N_r,), (M+1, K, N_r)
        case _:
            raise ValueError(f"Unknown method: {method}")

    f = func(r[:, None] * R, t[None, :])  # (N_r, N_theta)

    # RFFT along Theta (axis 1)
    F: np.ndarray = scipy.fft.rfft(f, axis=-1, overwrite_x=True) * dt  # (N_r, N_theta//2+1) # type: ignore
    F = F[:, : M + 1]  # (N_r, M+1)

    # DHT along R (axis 0): N_r -> K
    match method:
        case "legendre":
            coeffs = np.einsum("rm, r, mkr -> mk", F, r * w, J, optimize=True)  # (M+1, K)
        case "simpson":
            coeffs = simpson(np.einsum("rm, r, mkr -> rmk", F, r, J, optimize=True), x=r, axis=0)  # (M+1, K)

    coeffs /= norm_factors(M, K)  # (M+1, K)
    return coeffs


def inverse_transform_fast(coeffs, R, N=256, N_theta=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, K = coeffs.shape[0] - 1, coeffs.shape[1]
    N_r = int(np.hypot(N, N))
    N_theta = N_theta or scipy.fft.next_fast_len(int(2 * np.pi * N))  # type: ignore
    assert isinstance(N_theta, int)

    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)

    # IDHT along R (axis 1): K -> N_r
    r, J = bessel_basis(M, K, N_r)  # (N_r,), (M+1, K, N_r)
    C = np.einsum("mk, mkr -> mr", coeffs, J)  # (M+1, N_r)

    # IFFT along Theta (axis 0): M+1 -> N_theta using IRFFT
    f: np.ndarray = scipy.fft.irfft(C, n=N_theta, axis=0, overwrite_x=True) * N_theta  # (N_theta, N_r) # type: ignore
    f = np.pad(f, ((0, 1), (0, 0)), mode="wrap")  # (N_theta+1, N_r)

    # Interpolate Polar -> Cartesian
    Ys, Xs = np.mgrid[-R : R : N * 1j, -R : R : N * 1j]  # (N, N)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N, N)
    grid = np.stack([np.mod(Ts, 2 * np.pi), Rs], axis=-1)  # (N, N, 2)
    f = RegularGridInterpolator((t, r * R), f, bounds_error=False, fill_value=0)(grid)  # (N, N)

    return Xs, Ys, f


def fit(func, R, M, K, N_r=58, N_theta=None, activation=None, reg=None, init_coeffs=None, lr=1e-2, criterion=None, epochs=200, verbose=False, device=None):
    """
    Compute spectral coefficients by fitting reconstruction to sampled values, supporting custom post-processing (e.g. clipping).
    Uses gradient descent to optimize coefficients to minimize MSE after activation.

    Parameters
    ----------
    func : callable
        Function f(r, theta).
    R : float
        Disk radius.
    M : int
        Max angular order.
    K : int
        Number of radial modes.
    N_r, N_theta : int
        Grid size for sampling.
    activation : callable, optional
        Function to apply to the reconstructed values before loss calculation.
        Useful for handling clipping or other non-linearities.
    reg : callable, optional
        Regularization function on coefficients. Default is no regularization.
    init_coeffs : np.ndarray, optional
        Initial coefficients for optimization. If None, uses direct transform.
    lr : float
        Learning rate.
    criterion: callable, optional
        Loss function (pred, target). Default is MSE.
    epochs : int
        Number of optimization steps.
    verbose : bool
        Whether to print progress.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Defaults to 'cuda' if available.

    Returns
    -------
    coeffs : np.ndarray
        Optimized coefficients (M+1, K).
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    N_theta = N_theta or min_n_theta(M)

    # Sample on grid
    m = np.arange(0, M + 1)  # (M+1,)
    r, _, J = legendre_basis(M, K, N_r)  # (N_r,), (N_r,), (M+1, K, N_r)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)

    # Target values
    f_true = func(r[:, None] * R, t[None, :])  # (N_r, N_theta)

    E = np.exp(1j * t[:, None] * m[None, :])  # (N_theta, M+1)
    w = np.r_[1, [2] * M]

    if init_coeffs is None:
        init_coeffs = transform(func, R, M, K, N_r, N_theta, method="legendre")
    coeffs = torch.tensor(init_coeffs, dtype=torch.complex64, device=device, requires_grad=True)

    J = torch.from_numpy(J).to(device=device, dtype=torch.complex64)
    E = torch.from_numpy(E).to(device=device, dtype=torch.complex64)
    w = torch.from_numpy(w).to(device=device, dtype=torch.complex64)
    f_true = torch.from_numpy(f_true).to(device=device, dtype=torch.float32)

    criterion = criterion or torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam([coeffs], lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()

        f = torch.einsum("mk, mkr, tm, m -> rt", coeffs, J, E, w).real

        if activation:
            f = activation(f)

        loss = criterion(f, f_true)
        if reg:
            loss = loss + reg(coeffs)

        loss.backward()
        optimizer.step()

        if verbose and i % 10 == 0:
            print(f"Iter {i}: Loss {loss.item():.6e}")

    return coeffs.detach().cpu().numpy()


if __name__ == "__main__":
    import torch
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def latent_func(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z1 = 2.0 * np.exp(-((x - 80) ** 2 + y**2) / (2 * 100**2))
        z2 = -8.0 * np.exp(-((x + 80) ** 2 + (y - 50) ** 2) / (2 * 100**2))
        return (z1 + z2) * soft_mask(r, R=R)

    CLIP_MIN, CLIP_MAX = -2, None

    def observed_func(r, theta):
        return np.clip(latent_func(r, theta), CLIP_MIN, CLIP_MAX)

    R = 256
    M = 8
    K = 8
    N_R = 100
    N_XY = 200

    print("=" * 58)
    print(f"Disk Spec. Decomp. (R={R}, M={M}, K={K})")

    # Transform and Reconstruction
    c_dir = transform(observed_func, R, M, K, N_r=N_R)
    c_fit = fit(observed_func, R, M, K, N_r=N_R, activation=lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX), init_coeffs=c_dir, lr=0.01, epochs=200)
    c_fast = transform_fast(observed_func, R, M, K, N_r=N_R)
    Xs, Ys, f_rec = inverse_transform(c_dir, R, N=N_XY)
    Xs, Ys, f_fast = inverse_transform_fast(c_dir, R, N=N_XY)
    Xs, Ys, f_fit = inverse_transform(c_fit, R, N=N_XY)
    f_fit = np.clip(f_fit, CLIP_MIN, CLIP_MAX)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)
    f_lat = latent_func(Rs, Ts) * (Rs <= R)
    f_obs = observed_func(Rs, Ts) * (Rs <= R)

    def stats(name, d):
        print(f"{name:<25} | MaxAbs: {np.max(d):.1e} | MSE: {np.mean(d**2):.1e}")

    print("-" * 58)
    stats("Transform Diff (Fast-Std)", np.abs(c_dir - c_fast))
    stats("InvTrans Diff (Fast-Std)", np.abs(f_rec - f_fast))
    print("-" * 58)
    stats("Observed vs Direct", np.abs(f_obs - f_rec))
    stats("Observed vs Fitted", np.abs(f_obs - f_fit))
    print("-" * 58)

    # Visualization
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Observed (Clipped)",
            "Direct Recon",
            "Fitted Recon",
            "Latent Truth",
            "Direct Error",
            "Fitted Error",
        ),
        specs=[[{"type": "surface"}] * 3] * 2,
        vertical_spacing=0.03,
        horizontal_spacing=0.03,
    )

    def add_surf(r, c, vol, imin, imax, cmap="Plotly3", showscale=False):
        fig.add_trace(
            go.Surface(
                x=Xs,
                y=Ys,
                z=vol,
                cmin=imin,
                cmax=imax,
                colorscale=cmap,
                showscale=showscale,
            ),
            row=r,
            col=c,
        )

    fs = [f_obs, f_rec, f_fit, f_lat]
    v_min, v_max = min(f.min() for f in fs), max(f.max() for f in fs)
    max_err = np.abs(f_rec - f_obs).max()

    add_surf(1, 1, f_obs, v_min, v_max)
    add_surf(1, 2, f_rec, v_min, v_max)
    add_surf(1, 3, f_fit, v_min, v_max)
    add_surf(2, 1, f_lat, v_min, v_max)
    add_surf(2, 2, np.abs(f_rec - f_obs), 0, max_err, "Hot")
    add_surf(2, 3, np.abs(f_fit - f_obs), 0, max_err, "Hot")

    scene_common = dict(zaxis=dict(range=[v_min, v_max]))
    scene_err = dict(zaxis=dict(range=[0, max_err]))
    fig.update_layout(
        title="Disk Clipped Reconstruction Benchmark",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        scene=scene_common,
        scene2=scene_common,
        scene3=scene_common,
        scene4=scene_common,
        scene5=scene_err,
        scene6=scene_err,
    )
    fig.write_html("disk_decomposition.html", include_plotlyjs="cdn")
