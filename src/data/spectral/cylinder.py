from typing import Literal

import numpy as np
import scipy.fft
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d

from . import disk, interval


def soft_mask(r, z, R, L, margin_r=None, margin_z=None):
    """
    Create a soft mask on the cylinder domain that smoothly transitions from 1 to 0 near the boundaries.
    Use cosine tapering over the specified smooth widths in r and z directions.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinates.
    z : np.ndarray
        Longitudinal coordinates.
    R : float
        Radius of the cylinder.
    L : float
        Length of the cylinder.
    margin_r : float, optional
        Width over which the mask transitions from 1 to 0 at r = R.
    margin_z : float, optional
        Width over which the mask transitions from 1 to 0 at z = 0 and z = L.
    """
    M_r = disk.soft_mask(r, R=R, margin=margin_r)
    M_z = interval.soft_mask(z, L=L, margin=margin_z)
    return M_r * M_z


def transform(func, R, L, M, K_r, K_z, N_r=60, N_theta=None, N_z=60, method: Literal["legendre", "simpson", "dht"] = "legendre"):
    """
    Compute the Fourier-Bessel-Sine spectral coefficients of a real-valued function defined on a cylinder.

        **Domain**:
            Cylinder = B^2 x I = {(r, theta, z) | 0 <= r <= R, 0 <= theta < 2pi, 0 <= z <= L}

        **Basis**:
            Fourier-Bessel in (r, theta), Sine in z.
            J_m(alpha_mk * r / R) * exp(i * m * theta) * sin(l * pi * z / L)
            for m = 0, 1, ..., M; k = 1, 2, ..., K_r; l = 1, 2, ..., K_z,
            where alpha_mk is the k-th positive zero of J_m.

    Parameters
    ----------
    func : callable
        Function f(r, theta, z).
    R : float
        Radius.
    L : float
        Length.
    M : int
        Max angular order (0 <= m <= M).
    K_r : int
        Number of radial modes.
    K_z : int
        Number of longitudinal (z) modes.
    N_r : int
        Number of radial sample points.
    N_theta : int
        Number of angular sample points.
    N_z : int
        Number of longitudinal (z) sample points.
    method : str
        'legendre': Uses Gauss-Legendre quadrature for radial direction.
        'simpson': Uses Simpson's rule on uniform radial grid.
        'dht': Uses Discrete Hankel Transform (DHT) for radial direction.

    Returns
    -------
    coeffs : np.ndarray
        Shape (M+1, K_r, K_z).
    """
    N_theta = N_theta or disk.min_n_theta(M)
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    dz = 1 / N_z
    dt = 2 * np.pi / N_theta

    m = np.arange(0, M + 1)  # (M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)  # noqa: E741
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    z = interval.samples(N_z)  # (N_z,)

    E = np.exp(-1j * m[:, None] * t[None, :])  # (M+1, N_theta)
    S = np.sin(l[:, None] * np.pi * z[None, :])  # (K_z, N_z)
    match method:
        case "legendre":
            r, w, J = disk.legendre_basis(M, K_r, N_r)  # (N_r,), (N_r,), (M+1, K_r, N_r)
            f = func(r[:, None, None] * R, t[None, :, None], z[None, None, :] * L)  # (N_r, N_theta, N_z)
            coeffs = np.einsum("rtz, mt, lz, r, mkr -> mkl", f, E, S, r * w * dt * dz, J, optimize=True)  # (M+1, K_r, K_z)
        case "simpson":
            r, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
            f = func(r[:, None, None] * R, t[None, :, None], z[None, None, :] * L)  # (N_r, N_theta, N_z)
            integrand = np.einsum("rtz, mt, lz, r, mkr -> rmkl", f, E, S, r * dt * dz, J, optimize=True)  # (N_r, M+1, K_r, K_z)
            coeffs = simpson(integrand, x=r, axis=0)  # (M+1, K_r, K_z)
        case "dht":
            r, w, J = disk.dht_basis(M, K_r)  # (M+1, K_r), (M+1, K_r), (M+1, K_r, K_r)
            f = func(r[:, :, None, None] * R, t[None, None, :, None], z[None, None, None, :] * L)  # (M+1, K_r, N_theta, N_z)
            coeffs = np.einsum("mrtz, mt, lz, mkr, mr -> mkl", f, E, S, J, w * dt * dz, optimize=True)  # (M+1, K_r, K_z)
        case _:
            raise ValueError(f"Unknown method: {method}")

    coeffs /= disk.norm_factors(M, K_r)[:, :, None] / 2  # (M+1, K_r, K_z)
    return coeffs


def inverse_transform(coeffs, R, L, N_xy=100, N_z=100):
    """
    Reconstruct the function from its Fourier-Bessel-Sine spectral coefficients on a cylinder.

    Parameters
    ----------
    coeffs : np.ndarray
        (M+1, K_r, K_z)
    R, L : float
        Domain parameters.
    N_xy, N_z : int
        Output grid size.

    Returns
    -------
    Xs, Ys, Zs : np.ndarray
        The grid coordinates.
    f : np.ndarray
        The reconstructed function values on the 3D grid.
    """
    M, K_r, K_z = coeffs.shape[0] - 1, coeffs.shape[1], coeffs.shape[2]
    N_r = int(np.hypot(N_xy, N_xy))

    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N_xy, N_xy)

    m = np.arange(0, M + 1)  # (M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)  # noqa: E741
    z = interval.samples(N_z)  # (N_z,)

    Xs = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ys = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Zs = np.broadcast_to(z[:, None, None] * L, (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)

    r, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
    S = np.sin(l[:, None] * np.pi * z[None, :])  # (K_z, N_z)
    E = np.exp(1j * m[:, None, None] * Ts[None, :, :])  # (M+1, N_xy, N_xy)

    C = np.einsum("mkl, lz, mkr -> mzr", coeffs, S, J)  # (M+1, N_z, N_r)
    C = interp1d(r * R, C, axis=-1, bounds_error=False, fill_value=0)(Rs)  # (M+1, N_z, N_xy, N_xy)
    f = C[0].real * E[0].real + 2 * np.einsum("mzxy, mxy -> zxy", C[1:], E[1:], optimize=True).real  # (N_z, N_xy, N_xy)

    return Xs, Ys, Zs, f


def transform_fast(func, R, L, M, K_r, K_z, N_r=60, N_theta=200, N_z=60, method: Literal["legendre", "simpson"] = "legendre"):
    N_theta = N_theta or disk.min_n_theta(M)
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    dz = L / N_z
    dt = 2 * np.pi / N_theta

    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    z = interval.samples(N_z)  # (N_z,)

    match method:
        case "legendre":
            r, w, J = disk.legendre_basis(M, K_r, N_r)
        case "simpson":
            r, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
        case _:
            raise ValueError(f"Unknown method: {method}")

    f = func(r[:, None, None] * R, t[None, :, None], z[None, None, :] * L)  # (N_r, N_theta, N_z)

    # RFFT along Theta (axis 1): N_theta -> M+1
    F: np.ndarray = scipy.fft.rfft(f, axis=1, overwrite_x=True) * dt  # (N_r, N_theta//2+1, N_z) # type: ignore
    F = F[:, : M + 1, :]  # (N_r, M+1, N_z)

    # DST along Z (axis 2): N_z -> K_z
    F = scipy.fft.dst(F, axis=2, overwrite_x=True) * dz  # (N_r, M+1, K_z)
    F = F[:, :, :K_z]  # (N_r, M+1, K_z)

    # DHT along R (axis 0): N_r -> K_r
    if method == "legendre":
        coeffs = np.einsum("rml, r, r, mkr -> mkl", F, r, w, J, optimize=True)  # (M+1, K_r, K_z)
    else:  # simpson
        integrand = np.einsum("rml, r, mkr -> rmkl", F, r, J, optimize=True)  # (N_r, M+1, K_r, K_z)
        coeffs = simpson(integrand, x=r, axis=0)  # (M+1, K_r, K_z)
    coeffs /= disk.norm_factors(M, K_r)[:, :, None] * L  # (M+1, K_r, K_z)
    return coeffs


def inverse_transform_fast(coeffs, R, L, N_xy=100, N_z=100, N_r=None, N_theta=None):
    M, K_r, K_z = coeffs.shape[0] - 1, coeffs.shape[1], coeffs.shape[2]  # noqa: F841
    N_r = N_r or int(np.hypot(N_xy, N_xy))
    N_theta = N_theta or scipy.fft.next_fast_len(int(2 * np.pi * N_r))  # type: ignore
    assert isinstance(N_theta, int)

    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    z = interval.samples(N_z)  # (N_z,)

    # IDST along Z (axis 2): K_z -> N_z
    C = scipy.fft.idst(coeffs * N_z, axis=2, n=N_z, overwrite_x=True)  # (M+1, K_r, N_z) # type: ignore

    # IDHT along R (axis 1): K_r -> N_r
    r, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
    C: np.ndarray = np.einsum("mkz, mkr -> mrz", C, J)  # (M+1, N_r, N_z)

    # IRFFT along Theta (axis 0): M+1 -> N_theta using IRFFT
    f: np.ndarray = scipy.fft.irfft(C, n=N_theta, axis=0, overwrite_x=True) * N_theta  # (N_theta, N_r, N_z) # type: ignore
    f = np.pad(f, ((0, 1), (0, 0), (0, 0)), mode="wrap")  # (N_theta+1, N_r, N_z)

    # Interpolate Cylindrical -> Cartesian
    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N_xy, N_xy)
    Xs = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ys = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Zs = np.broadcast_to(z[:, None, None] * L, (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ts = np.broadcast_to(Ts[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Rs = np.broadcast_to(Rs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    grid = np.stack([np.mod(Ts, 2 * np.pi), Rs, Zs], axis=-1)  # (N_z, N_xy, N_xy, 3)
    f = RegularGridInterpolator((t, r * R, z * L), f, bounds_error=False, fill_value=0)(grid)  # (N_z, N_xy, N_xy)

    return Xs, Ys, Zs, f


def fit(func, R, L, M, K_r, K_z, N_r=60, N_theta=None, N_z=60, activation=None, reg=None, lr=1e-2, epochs=200, verbose=False, device=None):
    """
    Compute spectral coefficients by fitting reconstruction to sampled values, supporting custom post-processing (e.g. clipping).
    Uses gradient descent to optimize coefficients to minimize MSE after activation.

    Parameters
    ----------
    func : callable
        Function f(r, theta, z).
    R, L : float
        Cylinder dimensions.
    M : int
        Max angular order.
    K_r, K_z : int
        Number of modes.
    N_r, N_theta, N_z : int
        Grid size for sampling.
    activation : callable, optional
        Function to apply to the reconstructed values before loss calculation.
        Useful for handling clipping or other non-linearities.
    reg : callable, optional
        Regularization function on coefficients. Default is no regularization.
    lr : float
        Learning rate.
    epochs : int
        Number of optimization steps.

    Returns
    -------
    coeffs : np.ndarray
        Optimized coefficients (M+1, K_r, K_z).
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    N_theta = N_theta or disk.min_n_theta(M)

    # Sample on grid
    m = np.arange(0, M + 1)  # (M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)  # noqa: E741
    r, _, J = disk.legendre_basis(M, K_r, N_r)  # (N_r,), (N_r,), (M+1, K_r, N_r)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    z = interval.samples(N_z)

    # Target values
    f_true = func(r[:, None, None] * R, t[None, :, None], z[None, None, :] * L)  # (N_r, N_theta, N_z)

    E = np.exp(1j * t[:, None] * m[None, :])  # (N_theta, M+1)
    S = np.sin(l[None, :] * np.pi * z[:, None])  # (N_z, K_z)
    w = np.r_[1, [2] * M]

    init_coeffs = transform(func, R, L, M, K_r, K_z, N_r, N_theta, N_z, method="legendre")
    coeffs = torch.tensor(init_coeffs, dtype=torch.complex64, device=device, requires_grad=True)

    J = torch.from_numpy(J).to(device=device, dtype=torch.complex64)
    E = torch.from_numpy(E).to(device=device, dtype=torch.complex64)
    S = torch.from_numpy(S).to(device=device, dtype=torch.complex64)
    w = torch.from_numpy(w).to(device=device, dtype=torch.complex64)
    f_true = torch.from_numpy(f_true).to(device=device, dtype=torch.float32)

    def loss_fn(f):
        return torch.nn.functional.mse_loss(f, f_true)

    optimizer = torch.optim.Adam([coeffs], lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()

        f = torch.einsum("mkl, zl, mkr, tm, m -> rtz", coeffs, S, J, E, w).real

        if activation:
            f = activation(f)

        loss = loss_fn(f)
        if reg:
            loss = loss + reg(coeffs)

        loss.backward()

        optimizer.step()

        if verbose and i % 10 == 0:
            print(f"Iter {i}: Loss {loss.item():.6e}")

    return coeffs.detach().cpu().numpy()


if __name__ == "__main__":
    import torch
    from plotly.subplots import make_subplots

    from ...visualization.objects import VoxelIsosurface

    def latent_func(r, theta, z):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z1 = -2.0 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2 + (z - 50) ** 2) / (2 * 80**2))
        z2 = 10 * np.exp(-((x + 30) ** 2 + (y - 40) ** 2 + (z - 100) ** 2 / 2) / (2 * 20**2))
        return (z1 + z2) * soft_mask(r, z, R=R, L=L)

    CLIP_MIN, CLIP_MAX = None, 1

    def observed_func(r, theta, z):
        return np.clip(latent_func(r, theta, z), CLIP_MIN, CLIP_MAX)

    # Setup Domain
    R = 100
    L = 150
    M = 10
    K_R = 10
    K_Z = 10
    N_R = 40
    N_XY = 150
    N_Z = 120

    print("=" * 58)
    print(f"Cyl. Spec. Decomp. (R={R}, L={L}, M={M}, K={K_R}x{K_Z})")

    # Transform and Reconstruction
    c_dir = transform(observed_func, R, L, M, K_R, K_Z, N_R, N_z=N_Z)
    c_fit = fit(observed_func, R, L, M, K_R, K_Z, N_R, N_theta=100, N_z=N_Z, activation=lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX), lr=0.05, epochs=200)
    c_fast = transform_fast(observed_func, R, L, M, K_R, K_Z, N_R, N_z=N_Z)
    Xs, Ys, Zs, f_rec = inverse_transform(c_dir, R, L, N_XY, N_Z)
    Xs, Ys, Zs, f_fast = inverse_transform_fast(c_dir, R, L, N_XY, N_Z)
    Xs, Ys, Zs, f_fit = inverse_transform(c_fit, R, L, N_XY, N_Z)
    f_fit = np.clip(f_fit, CLIP_MIN, CLIP_MAX)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)
    f_lat = latent_func(Rs, Ts, Zs) * (Rs <= R)
    f_obs = observed_func(Rs, Ts, Zs) * (Rs <= R)

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
        specs=[[{"type": "volume"}] * 3] * 2,
        vertical_spacing=0.03,
        horizontal_spacing=0.03,
    )

    def add_iso(r, c, vol, imin, imax, cmap="Plotly3", showscale=False):
        fig.add_traces(
            VoxelIsosurface(
                x_range=(-R, R),
                y_range=(-R, R),
                z_range=(0, L),
                value=vol,
                isomin=imin,
                isomax=imax,
                surface_count=20,
                opacity=0.2,
                colorscale=cmap,
                showscale=showscale,
            ),
            rows=r,
            cols=c,
        )

    fs = [f_obs, f_rec, f_fit, f_lat]
    v_min, v_max = min(f.min() for f in fs), max(f.max() for f in fs)
    max_err = np.abs(f_rec - f_obs).max()

    add_iso(1, 1, f_obs, v_min, v_max)
    add_iso(1, 2, f_rec, v_min, v_max)
    add_iso(1, 3, f_fit, v_min, v_max)
    add_iso(2, 1, f_lat, v_min, v_max)
    add_iso(2, 2, np.abs(f_rec - f_obs), 0, max_err, "Hot", True)
    add_iso(2, 3, np.abs(f_fit - f_obs), 0, max_err, "Hot")
    fig.update_layout(
        template="plotly_dark",
        title="Clipped Reconstruction Benchmark",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    for i in range(1, 7):
        fig.layout[f"scene{i}" if i > 1 else "scene"].update(aspectratio=dict(x=1, y=1, z=L / (2 * R)))
    fig.write_html("cylinder_decomposition.html", include_plotlyjs="cdn")
