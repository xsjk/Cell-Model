from functools import cache
from typing import Literal

import numpy as np
import scipy.fft
import scipy.special as sp
import torch
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


def inverse_transform_fast(coeffs, R, N=256, N_r=None, N_theta=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, K = coeffs.shape[0] - 1, coeffs.shape[1]
    N_r = N_r or int(np.hypot(N, N))
    N_theta = N_theta or scipy.fft.next_fast_len(int(2 * np.pi * N_r))  # type: ignore
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


def transform_torch(data, M, K, N_r=None, N_theta=None, method: Literal["legendre", "simpson"] = "simpson"):
    assert data.ndim == 2 and data.shape[0] == data.shape[1]
    device, dtype = data.device, data.dtype
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    N = data.shape[0]
    N_r = N_r or N // 2
    N_theta = N_theta or min_n_theta(M)

    # Coordinates & Weights
    match method:
        case "legendre":
            r, w, J = legendre_basis(M, K, N_r)
            r = torch.from_numpy(r).to(device, dtype)
            w = torch.from_numpy(w).to(device, dtype)
        case "simpson":
            # Simpson's Rule weights: 1-4-2-4-2...4-1 on uniform grid [0, 1]
            r, J = bessel_basis(M, K, N_r)
            r = torch.from_numpy(r).to(device, dtype)
            w = torch.ones(N_r, device=device, dtype=dtype)
            w[1:-1:2], w[2:-1:2] = 4, 2
            w *= (1.0 / (N_r - 1)) / 3.0
        case _:
            raise ValueError(f"Unknown method: {method}")
    J = torch.from_numpy(J).to(device, cdtype)  # (M+1, K, N_r)

    # Resample Cartesian -> Polar
    t = torch.linspace(0, 2 * torch.pi, N_theta + 1, device=device, dtype=dtype)[:-1]
    grid = torch.empty((1, N_r, N_theta, 2), device=device, dtype=dtype)
    grid[0, ..., 0] = r[:, None] * torch.cos(t[None, :])
    grid[0, ..., 1] = r[:, None] * torch.sin(t[None, :])
    f = torch.nn.functional.grid_sample(data[None, None, ...], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]

    # Spectral Projections (FFT, Bessel)
    F = torch.fft.rfft(f, dim=1)[:, : M + 1] * (2 * torch.pi / N_theta)  # (N_r, M+1)
    coeffs = torch.einsum("rm, r, r, mkr -> mk", F, r, w, J)  # (M+1, K)

    # Normalization
    norms = torch.from_numpy(norm_factors(M, K)).to(device, cdtype)
    return coeffs / norms


def inverse_transform_torch(coeffs, N, N_r=None, N_theta=None):
    assert coeffs.ndim == 2
    device, cdtype = coeffs.device, coeffs.dtype
    dtype = torch.float64 if cdtype == torch.complex128 else torch.float32

    M, K = coeffs.shape[0] - 1, coeffs.shape[1]
    N_r = N_r or int(np.hypot(N, N))
    N_theta = N_theta or int(scipy.fft.next_fast_len(int(2 * np.pi * N_r)))  # type: ignore

    _, J = bessel_basis(M, K, N_r)  # (N_r,), (M+1, K, N_r)
    J = torch.from_numpy(J).to(device=device, dtype=cdtype)  # (M+1, K, N_r)

    # Project to Polar Grid
    C = torch.einsum("mk, mkr -> mr", coeffs, J)  # (M+1, N_r)
    f = torch.fft.irfft(C, n=N_theta, dim=0) * N_theta  # (N_theta, N_r)
    f = torch.cat([f, f[:1, :]], dim=0)  # (N_theta+1, N_r)

    # Coordinate Mapping (Normalized)
    Y, X = torch.meshgrid(*[torch.linspace(-1, 1, N, device=device, dtype=dtype)] * 2, indexing="ij")
    Rs, Ts = torch.hypot(X, Y), torch.atan2(Y, X) % (2 * torch.pi)

    grid = torch.empty((1, N, N, 2), device=device, dtype=dtype)
    grid[0, ..., 0] = Rs * 2 - 1
    grid[0, ..., 1] = (Ts / (2 * torch.pi)) * 2 - 1

    return torch.nn.functional.grid_sample(f[None, None, ...], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, ...]


def fit(data, M, K, N_r=None, N_theta=None, activation=None, reg=None, init_coeffs=None, lr=5e-2, criterion=None, epochs=1000, log_interval=None, device=None):
    """
    Fit spectral coefficients to a 2D data grid using gradient descent.
    """
    assert data.ndim == 2 and data.shape[0] == data.shape[1]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device=device)
    else:
        data = data.detach().to(device=device)
    data = data.detach()
    dtype, N = data.dtype, data.shape[0]
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    if init_coeffs is None:
        coeffs = transform_torch(data, M, K, N_r=N_r, N_theta=N_theta).requires_grad_(True)
    else:
        coeffs = torch.tensor(init_coeffs, dtype=cdtype, device=device, requires_grad=True)

    criterion = criterion or torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs)

    for i in range(epochs):
        optimizer.zero_grad()
        f = inverse_transform_torch(coeffs, N, N_r=N_r, N_theta=N_theta)
        if activation:
            f = activation(f)

        loss = criterion(f, data)
        if reg:
            loss = loss + reg(coeffs)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if log_interval is not None and i % log_interval == 0:
            print(f"Fit Iter {i}: Loss {loss.item():.6e}")

    with torch.inference_mode():
        f = inverse_transform_torch(coeffs, N, N_r=N_r, N_theta=N_theta)
        if activation:
            f = activation(f)

    return coeffs.detach().cpu().numpy(), f.detach().cpu().numpy()


if __name__ == "__main__":
    import plotly.graph_objects as go
    import torch
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transform and Reconstruction
    c_dir = transform(observed_func, R, M, K, N_r=N_R)
    c_fast = transform_fast(observed_func, R, M, K, N_r=N_R)
    Xs, Ys, f_rec = inverse_transform(c_dir, R, N=N_XY)
    Xs, Ys, f_fast = inverse_transform_fast(c_dir, R, N=N_XY)
    f_torch = inverse_transform_torch(torch.from_numpy(c_dir).to(device), N_XY).cpu().numpy()
    np.testing.assert_allclose(f_torch, f_fast, atol=1e-14)

    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)
    f_lat = latent_func(Rs, Ts) * (Rs <= R)
    f_obs = observed_func(Rs, Ts) * (Rs <= R)

    print("Fitting...")
    activation = lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX)  # noqa: E731
    c_fit, f_fit_torch = fit(f_obs, M, K, N_r=N_R, activation=activation, init_coeffs=c_dir, log_interval=100)
    Xs, Ys, f_fit_raw = inverse_transform_fast(c_fit, R, N=N_XY, N_r=N_R)
    f_fit = np.clip(f_fit_raw, CLIP_MIN, CLIP_MAX)
    np.testing.assert_allclose(f_fit_torch, f_fit, atol=1e-14)

    def stats(name, d):
        print(f"{name:<25} | MaxAbs: {np.max(d):.1e} | MSE: {np.mean(d**2):.1e}")

    print("-" * 58)
    stats("Transform Diff (Fast-Std)", np.abs(c_dir - c_fast))
    stats("InvTrans Diff (Fast-Std)", np.abs(f_rec - f_fast))
    print("-" * 58)
    stats("Observed vs Direct", np.abs(f_obs - f_rec))
    stats("Observed vs Fitted", np.abs(f_obs - f_fit_torch))
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
