from typing import Literal

import numpy as np
import plotly.graph_objects as go
import scipy.fft
from plotly.subplots import make_subplots
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


if __name__ == "__main__":

    def example_func(r, theta, z):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z1 = np.exp(-((x - 20) ** 2 + (y - 20) ** 2 + (z - 50) ** 2) / (2 * 20**2))
        z2 = -0.8 * np.exp(-((x + 30) ** 2 + (y - 40) ** 2 + (z - 100) ** 2 / 2) / (2 * 15**2))
        return (z1 + z2) * soft_mask(r, z, R=R, L=L)

    R = 100
    L = 150
    M = 15  # angular orders
    K_R = 10  # radial modes
    K_Z = 10  # longitudinal modes
    N_R = 50
    N_XY = 60
    N_Z = 45

    # Transform
    coeffs = transform(example_func, R=R, L=L, M=M, K_r=K_R, K_z=K_Z, N_r=N_R, N_z=N_Z)
    coeffs_fast = transform_fast(example_func, R=R, L=L, M=M, K_r=K_R, K_z=K_Z, N_r=N_R, N_z=N_Z)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    Xs, Ys, Zs, Vol_rec = inverse_transform(coeffs, R=R, L=L, N_xy=N_XY, N_z=N_Z)
    Xs_f, Ys_f, Zs_f, Vol_rec_fast = inverse_transform_fast(coeffs, R=R, L=L, N_xy=N_XY, N_z=N_Z)
    print(f"Inverse Transform max diff: {np.max(np.abs(Vol_rec - Vol_rec_fast)):.6e}")

    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)

    # Get ground truth
    Vol_true = example_func(Rs, Ts, Zs)
    Vol_true[Rs > R] = 0

    # Display errors
    print("\nError vs Ground Truth:")
    abs_err = np.abs(Vol_true - Vol_rec)
    max_abs_err = np.max(abs_err)
    print(f"MAE: {max_abs_err:.6e}")
    print(f"RMSE: {np.sqrt(np.mean((Vol_true - Vol_rec) ** 2)):.6e}")

    # Visualization
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "volume"}, {"type": "volume"}, {"type": "volume"}]],
        subplot_titles=(
            "Original Cylinder Function",
            "Reconstructed",
            "Absolute Error",
        ),
    )

    fig.add_trace(
        go.Volume(
            x=Xs.flatten(),
            y=Ys.flatten(),
            z=Zs.flatten(),
            value=Vol_true.flatten(),
            isomin=-0.8,
            isomax=0.8,
            opacity=0.2,
            surface_count=30,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Volume(
            x=Xs.flatten(),
            y=Ys.flatten(),
            z=Zs.flatten(),
            value=Vol_rec.flatten(),
            isomin=-0.8,
            isomax=0.8,
            opacity=0.2,
            surface_count=30,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Plot the error slices through the max error point
    max_error_idx = np.unravel_index(np.argmax(np.abs(Vol_true - Vol_rec)), Vol_true.shape)
    for dim, idx in enumerate(max_error_idx):
        Xs_slice = np.take(Xs, idx, axis=dim)
        Ys_slice = np.take(Ys, idx, axis=dim)
        Zs_slice = np.take(Zs, idx, axis=dim)
        abs_err_slice = np.take(abs_err, idx, axis=dim)
        fig.add_trace(
            go.Surface(
                x=Xs_slice,
                y=Ys_slice,
                z=Zs_slice,
                surfacecolor=abs_err_slice,
                colorscale="Hot",
                cmin=0.1 * max_abs_err,
                cmax=max_abs_err,
                showscale=False,  # only show color scale on volume plot
                name="Error Slice",
                opacity=0.2,
            ),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Volume(
            x=Xs.flatten(),
            y=Ys.flatten(),
            z=Zs.flatten(),
            value=abs_err.flatten(),
            isomin=0.1 * max_abs_err,
            isomax=max_abs_err,
            opacity=0.2,
            surface_count=30,
            colorscale="Hot",
            showscale=True,
            colorbar=dict(title="Abs Error"),
        ),
        row=1,
        col=3,
    )

    scene_cam = dict(aspectratio=dict(x=1, y=1, z=L / (R * 2)))
    fig.update_layout(
        title="Cylinder Spectral Decomposition (B^2 x I)",
        template="plotly_dark",
        scene=scene_cam,
        scene2=scene_cam,
        scene3=scene_cam,
    )

    output_file = "cylinder_decomposition.html"
    fig.write_html(output_file)
