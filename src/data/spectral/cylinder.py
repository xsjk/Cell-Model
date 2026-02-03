from typing import Literal

import numpy as np
import scipy.fft
import torch
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
    dz, dt = L / N_z, 2 * np.pi / N_theta
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    z = interval.samples(N_z)  # (N_z,)

    if method == "legendre":
        r, w, J = disk.legendre_basis(M, K_r, N_r)
    else:
        r, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)

    f = func(r[:, None, None] * R, t[None, :, None], z[None, None, :] * L)  # (N_r, N_theta, N_z)

    # Fast Spectral Projection (FFT, DST, Bessel)
    F = scipy.fft.rfft(f, axis=1)[:, : M + 1, :] * dt  # (N_r, M+1, N_z) # type: ignore[no-untyped-call]
    F = scipy.fft.dst(F, axis=2)[:, :, :K_z] * dz  # (N_r, M+1, K_z) # type: ignore[no-untyped-call]

    if method == "legendre":
        coeffs = np.einsum("rml, r, r, mkr -> mkl", F, r, w, J)
    else:
        coeffs = simpson(np.einsum("rml, r, mkr -> rmkl", F, r, J), x=r, axis=0)

    norms = L * disk.norm_factors(M, K_r)  # (M+1, K_r)
    return coeffs / norms[:, :, None]


def inverse_transform_fast(coeffs, R, L, N_xy=100, N_z=100, N_r=None, N_theta=None):
    M, K_r, K_z = coeffs.shape[0] - 1, coeffs.shape[1], coeffs.shape[2]  # noqa: F841
    N_r = N_r or int(np.hypot(N_xy, N_xy))
    N_theta = N_theta or int(scipy.fft.next_fast_len(int(2 * np.pi * N_r)))  # type: ignore

    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    r = np.linspace(0, 1, N_r)
    z = interval.samples(N_z)  # (N_z,)

    # Fast Inverse Projection (IDST, Bessel, IRFFT)
    C: np.ndarray = scipy.fft.idst(coeffs * N_z, axis=2, n=N_z)  # (M+1, K_r, N_z) # type: ignore[no-untyped-call]
    _, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
    C = np.einsum("mkz, mkr -> mrz", C, J)  # (M+1, N_r, N_z)
    f: np.ndarray = scipy.fft.irfft(C, n=N_theta, axis=0) * N_theta  # (N_theta, N_r, N_z) # type: ignore[no-untyped-call]
    f = np.pad(f, ((0, 1), (0, 0), (0, 0)), mode="wrap")  # (N_theta+1, N_r, N_z)

    # Interpolate to Cartesian
    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.mod(np.arctan2(Ys, Xs), 2 * np.pi)  # (N_xy, N_xy)
    Zs, Rs, Ts = np.broadcast_arrays(z[:, None, None] * L, Rs[None, ...], Ts[None, ...])
    grid = np.stack([Ts, Rs, Zs], axis=-1)  # (N_z, N_xy, N_xy, 3)
    f = RegularGridInterpolator((t, r * R, z * L), f, bounds_error=False, fill_value=0)(grid)
    return Xs, Ys, Zs, f


def transform_torch(data, M, K_r, K_z, N_r=None, N_theta=None, method: Literal["legendre", "simpson"] = "legendre"):
    assert data.ndim == 3 and data.shape[1] == data.shape[2]
    device, dtype = data.device, data.dtype
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    N_z, N_xy, _ = data.shape
    N_r = N_r or N_xy // 2
    N_theta = N_theta or disk.min_n_theta(M)

    # Coordinates & Weights
    match method:
        case "legendre":
            r, w, J = disk.legendre_basis(M, K_r, N_r)
            r = torch.from_numpy(r).to(device, dtype)
            w = torch.from_numpy(w).to(device, dtype)
        case "simpson":
            # Simpson's Rule weights: 1-4-2-4-2...4-1 on uniform grid [0, 1]
            r, J = disk.bessel_basis(M, K_r, N_r)
            r = torch.from_numpy(r).to(device, dtype)
            w = torch.ones(N_r, device=device, dtype=dtype)
            w[1:-1:2], w[2:-1:2] = 4, 2
            w *= (1.0 / (N_r - 1)) / 3.0
        case _:
            raise ValueError(f"Unknown method: {method}")
    J = torch.from_numpy(J).to(device, cdtype)  # (M+1, K_r, N_r)

    # Resample Cartesian -> Cylindrical
    t = torch.linspace(0, 2 * torch.pi, N_theta + 1, device=device, dtype=dtype)[:-1]
    z = (torch.arange(N_z, device=device, dtype=dtype) + 0.5) / N_z
    grid = torch.empty((1, N_r, N_theta, N_z, 3), device=device, dtype=dtype)
    grid[0, ..., 0] = r[:, None, None] * torch.cos(t[None, :, None])
    grid[0, ..., 1] = r[:, None, None] * torch.sin(t[None, :, None])
    grid[0, ..., 2] = (z[None, None, :] - 0.5 / N_z) / (1 - 1.0 / N_z) * 2 - 1
    f = torch.nn.functional.grid_sample(data[None, None, ...], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]

    # Spectral Projections (FFT, Sine, Bessel)
    F = torch.fft.rfft(f, dim=1)[:, : M + 1, :] * (2 * torch.pi / N_theta)  # (N_r, M+1, N_z)
    k = torch.arange(1, K_z + 1, device=device, dtype=dtype)  # (K_z,)
    S = torch.sin(k[:, None] * torch.pi * z[None, :]).to(cdtype)  # (K_z, N_z)
    coeffs = torch.einsum("rmz, lz, r, r, mkr -> mkl", F, S, r, w, J) / N_z  # (M+1, K_r, K_z)

    # Normalization
    norms = torch.from_numpy(disk.norm_factors(M, K_r)).to(device, cdtype)
    return coeffs / (norms[:, :, None] / 2)


def inverse_transform_torch(coeffs, N_xy, N_z, N_r=None, N_theta=None, bounds_z=(0, 1)):
    assert coeffs.ndim == 3
    device, cdtype = coeffs.device, coeffs.dtype
    dtype = torch.float64 if cdtype == torch.complex128 else torch.float32

    M, K_r, K_z = coeffs.shape[0] - 1, coeffs.shape[1], coeffs.shape[2]
    N_r = N_r or int(np.hypot(N_xy, N_xy))
    N_theta = N_theta or int(scipy.fft.next_fast_len(int(2 * np.pi * N_r)))  # type: ignore

    # Basis functions
    a, b = bounds_z
    z = a + (b - a) * (torch.arange(N_z, device=device, dtype=dtype) + 0.5) / N_z  # (N_z,)
    k = torch.arange(1, K_z + 1, device=device, dtype=dtype)  # (K_z,)
    S = torch.sin(k[:, None] * torch.pi * z[None, :]).to(cdtype)  # (K_z, N_z)
    _, J = disk.bessel_basis(M, K_r, N_r)  # (N_r,), (M+1, K_r, N_r)
    J = torch.from_numpy(J).to(device=device, dtype=cdtype)  # (M+1, K_r, N_r)

    # Project to Cylindrical Grid
    C = torch.einsum("mkl, lz, mkr -> mrz", coeffs, S, J)  # (M+1, N_r, N_z)
    f = torch.fft.irfft(C, n=N_theta, dim=0) * N_theta  # (N_theta, N_r, N_z)
    f = torch.cat([f, f[:1, :, :]], dim=0)  # (N_theta+1, N_r, N_z)

    # Coordinate Mapping (Normalized)
    Y, X = torch.meshgrid(*[torch.linspace(-1, 1, N_xy, device=device, dtype=dtype)] * 2, indexing="ij")
    Rs, Ts = torch.hypot(X, Y), torch.atan2(Y, X) % (2 * torch.pi)

    z_samp = (torch.arange(N_z, device=device, dtype=dtype) + 0.5) / N_z
    grid = torch.empty((1, N_z, N_xy, N_xy, 3), device=device, dtype=dtype)
    grid[0, ..., 0] = (z_samp[:, None, None] - 0.5 / N_z) / (1 - 1.0 / N_z) * 2 - 1
    grid[0, ..., 1] = Rs[None, :, :] * 2 - 1
    grid[0, ..., 2] = (Ts[None, :, :] / (2 * torch.pi)) * 2 - 1

    return torch.nn.functional.grid_sample(f[None, None, ...], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, ...]


def fit(
    data,
    M,
    K_r,
    K_z,
    N_r=None,
    N_theta=None,
    activation=None,
    reg=None,
    init_coeffs=None,
    lr=5e-2,
    criterion=None,
    epochs=1000,
    log_interval=None,
    device=None,
    bounds_z=(0, 1),
):
    """
    Fit spectral coefficients to a 3D data grid using gradient descent.
    """
    assert data.ndim == 3 and data.shape[1] == data.shape[2]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device=device)
    else:
        data = data.detach().to(device=device)
    data = data.detach()
    dtype, (N_z, N_xy, _) = data.dtype, data.shape
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

    if init_coeffs is None:
        coeffs = transform_torch(data, M, K_r, K_z, N_r=N_r, N_theta=N_theta).requires_grad_(True)
    else:
        coeffs = torch.tensor(init_coeffs, dtype=cdtype, device=device, requires_grad=True)

    criterion = criterion or torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs)

    for i in range(epochs):
        optimizer.zero_grad()
        f = inverse_transform_torch(coeffs, N_xy, N_z, N_r=N_r, N_theta=N_theta, bounds_z=bounds_z)
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
        f = inverse_transform_torch(coeffs, N_xy, N_z, N_r=N_r, N_theta=N_theta, bounds_z=bounds_z)
        if activation:
            f = activation(f)

    return coeffs.detach().cpu().numpy(), f.detach().cpu().numpy()


if __name__ == "__main__":
    import torch
    from plotly.subplots import make_subplots
    from plotly import graph_objects as go

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Setup
    TRUE_A_Z, TRUE_B_Z = 0.4, 0.8
    print(f"Observation interval (z): [{TRUE_A_Z:.2f}, {TRUE_B_Z:.2f}]")

    x = np.linspace(-R, R, N_XY)
    Xs, Ys = np.meshgrid(x, x, indexing="ij")
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)
    z_full = interval.samples(N_Z) * L
    z_obs = (TRUE_A_Z + (TRUE_B_Z - TRUE_A_Z) * interval.samples(N_Z)) * L
    f_lat_full = latent_func(Rs[None, :, :], Ts[None, :, :], z_full[:, None, None]) * (Rs <= R)[None, :, :]
    f_obs_full = observed_func(Rs[None, :, :], Ts[None, :, :], z_full[:, None, None]) * (Rs <= R)[None, :, :]
    f_obs_truncated = observed_func(Rs[None, :, :], Ts[None, :, :], z_obs[:, None, None]) * (Rs <= R)[None, :, :]

    # Transform and Reconstruction
    ## Direct Transform (treating truncated slices as full [0, 1] relative depth)
    c_dir = transform_torch(torch.from_numpy(f_obs_truncated).to(device), M, K_R, K_Z).cpu().numpy()
    _, _, _, f_rec_dir = inverse_transform_fast(c_dir, R, L, N_XY, N_Z)

    print("Fitting Truncated Data with Manual Bounds...")
    activation = lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX) if CLIP_MAX is not None else x  # noqa: E731
    c_fit, f_fit_slice = fit(f_obs_truncated, M, K_R, K_Z, activation=activation, bounds_z=(TRUE_A_Z, TRUE_B_Z), log_interval=100)

    # Reconstruct FULL volume from c_fit to see extrapolation/interpolation
    Xs, Ys, Zs, f_fit_full = inverse_transform_fast(c_fit, R, L, N_XY, N_Z)
    f_fit_full = np.clip(f_fit_full, CLIP_MIN, CLIP_MAX) if CLIP_MAX is not None else f_fit_full

    def stats(name, d):
        print(f"{name:<25} | MaxAbs: {np.max(d):.1e} | MSE: {np.mean(d**2):.1e}")

    print("-" * 58)
    stats("Observed vs Direct", np.abs(f_obs_truncated - f_rec_dir))
    stats("Observed vs Fitted", np.abs(f_obs_truncated - f_fit_slice))
    print("-" * 58)

    # Visualization
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            f"Observed Z=[{TRUE_A_Z},{TRUE_B_Z}]",
            "Direct (Fixed Z=[0,1])",
            "Fitted (Manual [a,b])",
            "Latent Full Z=[0,1]",
            "Direct Error (Trunc.)",
            "Fitted Error (Manual)",
        ),
        specs=[[{"type": "volume"}] * 3] * 2,
        vertical_spacing=0.03,
        horizontal_spacing=0.03,
    )

    def add_iso(r, c, vol, imin, imax, z_range=(0, L), cmap="Plotly3", showscale=False):
        fig.add_traces(
            VoxelIsosurface(
                x_range=(-R, R),
                y_range=(-R, R),
                z_range=z_range,
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

    def add_clip_plane(r, c, zp):
        fig.add_trace(
            go.Surface(
                x=[-R, R],
                y=[-R, R],
                z=[[zp, zp], [zp, zp]],
                opacity=0.2,
                showscale=False,
                colorscale=[[0, "cyan"], [1, "cyan"]],
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )

    v_min, v_max = f_lat_full.min(), f_lat_full.max()
    max_err = np.max(np.abs(f_obs_truncated - f_rec_dir))

    z_range = (TRUE_A_Z * L, TRUE_B_Z * L)
    add_iso(1, 1, f_obs_truncated, v_min, v_max, z_range=z_range)
    add_iso(1, 2, f_rec_dir, v_min, v_max, z_range=z_range)
    add_iso(1, 3, f_fit_full, v_min, v_max, z_range=(0, L))
    add_clip_plane(1, 3, TRUE_A_Z * L)
    add_clip_plane(1, 3, TRUE_B_Z * L)
    add_iso(2, 1, f_lat_full, v_min, v_max)
    add_clip_plane(2, 1, TRUE_A_Z * L)
    add_clip_plane(2, 1, TRUE_B_Z * L)
    add_iso(2, 2, np.abs(f_rec_dir - f_obs_truncated), 0, max_err, z_range=z_range, cmap="Hot", showscale=True)
    add_iso(2, 3, np.abs(f_fit_slice - f_obs_truncated), 0, max_err, z_range=z_range, cmap="Hot")
    fig.update_layout(
        template="plotly_dark",
        title="Cylindrical Spectral Fitting (Z-Truncation Benchmark)",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=L / (2 * R)),
        zaxis=dict(range=[0, L]),
    )
    fig.write_html("cylinder_decomposition.html", include_plotlyjs="cdn")
