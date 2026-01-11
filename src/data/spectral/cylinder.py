from typing import Literal

import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.special as sp
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d

from .disk import get_bessel_basis, get_legendre_basis, get_bessel_norm_factors, get_alphas


def transform(func, R, L, M, K_r, K_z, N_r=60, N_theta=200, N_z=60, method: Literal["legendre", "simpson"] = "legendre"):
    """
    Compute the Fourier-Bessel-Sine spectral coefficients of a function defined on a cylinder.
    Domain: Disk of radius R x Interval [0, L].

    Parameters
    ----------
    func : callable
        Function f(r, theta, z).
    R : float
        Radius.
    L : float
        Length.
    M : int
        Max angular order (|m| <= M).
    K_r : int
        Number of radial modes.
    K_z : int
        Number of longitudinal (z) modes.
    method : str
        'legendre': Uses Gauss-Legendre quadrature for radial direction (Spectral accuracy).
        'simpson': Uses Simpson's rule on uniform radial grid.

    Returns
    -------
    coeffs : np.ndarray
        Shape (2M+1, K_r, K_z).
    """
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    dz = L / N_z
    dt = 2 * np.pi / N_theta

    m = np.arange(-M, M + 1)  # (2M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)  # noqa: E741
    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)

    if method == "legendre":
        r_norm, w_norm, J = get_legendre_basis(M, K_r, N_r)
        r = R * r_norm  # (N_r,)
        w = R * w_norm  # (N_r,)
    elif method == "simpson":
        r = np.linspace(0, R, N_r)  # (N_r,)
        J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    else:
        raise ValueError(f"Unknown method: {method}")

    f = func(r[:, None, None], t[None, :, None], z[None, None, :])  # (N_r, N_theta, N_z)

    # FFT along Theta (axis 1): N_theta -> 2M+1
    E = np.exp(-1j * m[:, None] * t[None, :])  # (2M+1, N_theta)

    # DST along Z (axis 2): N_z -> K_z
    S = np.sin(l[:, None] * np.pi * z[None, :] / L)  # (K_z, N_z)

    # DHT along R (axis 0): N_r -> K_r
    F = np.einsum("rtz, mt, lz -> rml", f, E, S, optimize=True) * dt * dz  # (N_r, 2M+1, K_z)
    if method == "legendre":
        coeffs = np.einsum("rml, r, r, mkr -> mkl", F, r, w, J, optimize=True)  # (2M+1, K_r, K_z)
    else:  # simpson
        integrand = np.einsum("rml, r, mkr -> rmkl", F, r, J, optimize=True)  # (N_r, 2M+1, K_r, K_z)
        coeffs = simpson(integrand, x=r, axis=0)  # (2M+1, K_r, K_z)
    coeffs /= get_bessel_norm_factors(M, K_r)[:, :, None] * L / 2 * R**2  # (2M+1, K_r, K_z)
    return coeffs


def inverse_transform(coeffs, R, L, N_xy=100, N_z=100):
    """
    Reconstruct function from coefficients on a grid.

    Parameters
    ----------
    coeffs : np.ndarray
        (2M+1, K_r, K_z)
    R, L : float
        Domain parameters.
    N_xy, N_z : int
        Output grid size.

    Returns
    -------
    Xs, Ys, Zs : np.ndarray
    f : np.ndarray
        The reconstructed function values on the 3D grid.
    """
    M, K_r, K_z = (coeffs.shape[0] - 1) // 2, coeffs.shape[1], coeffs.shape[2]
    N_r = int(np.hypot(N_xy, N_xy))
    dz = L / N_z

    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N_xy, N_xy)

    m = np.arange(-M, M + 1)  # (2M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)  # noqa: E741
    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)
    r = np.linspace(0, R, N_r)  # (N_r,)

    Xs = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ys = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Zs = np.broadcast_to(z[:, None, None], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)

    S = np.sin(l[:, None] * np.pi * z[None, :] / L)  # (K_z, N_z)
    J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    E = np.exp(1j * m[:, None, None] * Ts[None, :, :])  # (2M+1, N_xy, N_xy)

    C = np.einsum("mkl, lz, mkr -> mrz", coeffs, S, J)  # (2M+1, N_r, N_z)
    C = np.moveaxis(C, 1, -1)  # (2M+1, N_z, N_r)
    C = interp1d(r, C, axis=-1, bounds_error=False, fill_value=0)(Rs)  # (2M+1, N_z, N_xy, N_xy)

    f = np.einsum("mzxy, mxy -> zxy", C, E, optimize=True)  # (N_z, N_xy, N_xy)

    return Xs, Ys, Zs, np.real(f)


def transform_fast(func, R, L, M, K_r, K_z, N_r=60, N_theta=200, N_z=60, method: Literal["legendre", "simpson"] = "legendre"):
    if N_theta < 2 * M + 1:
        raise ValueError("N_theta must be at least 2M + 1.")

    dz = L / N_z
    dt = 2 * np.pi / N_theta

    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)
    t = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    m = np.arange(-M, M + 1)  # (2M+1,)

    if method == "legendre":
        r_norm, w_norm, J = get_legendre_basis(M, K_r, N_r)
        Rs = R * r_norm  # (N_r,)
        Ws = R * w_norm  # (N_r,)
    elif method == "simpson":
        Rs = np.linspace(0, R, N_r)  # (N_r,)
        J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    else:
        raise ValueError(f"Unknown method: {method}")

    f = func(Rs[:, None, None], t[None, :, None], z[None, None, :])  # (N_r, N_theta, N_z)

    # FFT along Theta (axis 1): N_theta -> 2M+1
    F: np.ndarray = scipy.fft.fft(f, axis=1) * dt  # (N_r, 2M+1, N_z) # type: ignore
    F = F[:, m, :]  # (N_r, 2M+1, N_z)

    # DST along Z (axis 2): N_z -> K_z
    F = scipy.fft.dst(F, axis=2) * dz  # (N_r, 2M+1, K_z)
    F = F[:, :, :K_z]  # (N_r, 2M+1, K_z)

    # DHT along R (axis 0): N_r -> K_r
    if method == "legendre":
        coeffs = np.einsum("rml, r, r, mkr -> mkl", F, Rs, Ws, J, optimize=True)  # (2M+1, K_r, K_z)
    else:  # simpson
        integrand = np.einsum("rml, r, mkr -> rmkl", F, Rs, J, optimize=True)  # (N_r, 2M+1, K_r, K_z)
        coeffs = simpson(integrand, x=Rs, axis=0)  # (2M+1, K_r, K_z)
    coeffs /= get_bessel_norm_factors(M, K_r)[:, :, None] * L * R**2  # (2M+1, K_r, K_z)
    return coeffs


def inverse_transform_fast(coeffs, R, L, N_xy=100, N_z=100, N_r=None, N_theta=None):
    M, K_r, K_z = (coeffs.shape[0] - 1) // 2, coeffs.shape[1], coeffs.shape[2]  # noqa: F841
    N_r = N_r or int(np.hypot(N_xy, N_xy))
    N_theta = N_theta or int(2 * np.pi * N_xy)

    dz = L / N_z
    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    r = np.linspace(0, R, N_r)  # (N_r,)
    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)

    # IDST along Z (axis 2): K_z -> N_z
    C = scipy.fft.idst(coeffs * N_z, axis=2, n=N_z)  # (2M+1, K_r, N_z) # type: ignore

    # IDHT along R (axis 1): K_r -> N_r
    J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    C: np.ndarray = np.einsum("mkz, mkr -> mrz", C, J)  # (2M+1, N_r, N_z)

    # IFFT along Theta (axis 0): 2M+1 -> N_theta
    F = np.zeros((N_theta, N_r, N_z), dtype=np.complex128)  # (N_theta, N_r, N_z)
    F[m] = C
    f: np.ndarray = scipy.fft.ifft(F, axis=0).real * N_theta  # (N_theta, N_r, N_z) # type: ignore
    f = np.pad(f, ((0, 1), (0, 0), (0, 0)), mode="wrap")  # (N_theta+1, N_r, N_z)

    # Interpolate Cylindrical -> Cartesian
    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N_xy, N_xy)
    Xs = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ys = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Zs = np.broadcast_to(z[:, None, None], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Ts = np.broadcast_to(Ts[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    Rs = np.broadcast_to(Rs[None, ...], (N_z, N_xy, N_xy))  # (N_z, N_xy, N_xy)
    grid = np.stack([np.mod(Ts, 2 * np.pi), Rs, Zs], axis=-1)  # (N_z, N_xy, N_xy, 3)
    f = RegularGridInterpolator((t, r, z), f, bounds_error=False, fill_value=0)(grid)  # (N_z, N_xy, N_xy)

    return Xs, Ys, Zs, f


if __name__ == "__main__":

    def example_func(r, theta, z):
        # Gaussians in 3D
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Two Gaussian blobs in [-R,R]x[-R,R]x[0,L]
        z1 = np.exp(-((x - 20) ** 2 + (y - 20) ** 2 + (z - 50) ** 2) / (2 * 10**2))
        z2 = -0.8 * np.exp(-((x + 30) ** 2 + (y - 40) ** 2 + (z - 150) ** 2) / (2 * 15**2))

        return z1 + z2

    R = 100
    L = 200
    M_MAX = 10  # angular orders
    K_R = 20  # radial modes
    K_Z = 20  # longitudinal modes
    N_R = 50
    N_THETA = 30
    N_Z = 50

    # Transform
    coeffs = transform(example_func, R=R, L=L, M=M_MAX, K_r=K_R, K_z=K_Z, N_r=N_R, N_theta=N_THETA, N_z=N_Z)
    coeffs_fast = transform_fast(example_func, R=R, L=L, M=M_MAX, K_r=K_R, K_z=K_Z, N_r=N_R, N_theta=N_THETA, N_z=N_Z)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    Xs, Ys, Zs, Vol_rec = inverse_transform(coeffs, R=R, L=L, N_xy=N_R * 2, N_z=N_Z)
    Xs_f, Ys_f, Zs_f, Vol_rec_fast = inverse_transform_fast(coeffs, R=R, L=L, N_xy=N_R * 2, N_z=N_Z)
    print(f"Inverse Transform max diff: {np.max(np.abs(Vol_rec - Vol_rec_fast)):.6e}")

    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)

    # Get ground truth
    Vol_true = example_func(Rs, Ts, Zs)
    Vol_true[Rs > R] = 0

    # Display errors
    print("\nError vs Ground Truth:")
    print(f"MAE: {np.max(np.abs(Vol_true - Vol_rec)):.6e}")
    print(f"RMSE: {np.sqrt(np.mean((Vol_true - Vol_rec) ** 2)):.6e}")

    # Visualization
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "volume"}, {"type": "volume"}]],
        subplot_titles=(
            "Original Cylinder Function",
            "Reconstructed",
        ),
    )

    sample = np.s_[::2, ::2, ::2]

    fig.add_trace(
        go.Volume(
            x=Xs[sample].flatten(),
            y=Ys[sample].flatten(),
            z=Zs[sample].flatten(),
            value=Vol_true[sample].flatten(),
            isomin=-0.8,
            isomax=0.8,
            opacity=0.1,
            surface_count=30,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Volume(
            x=Xs[sample].flatten(),
            y=Ys[sample].flatten(),
            z=Zs[sample].flatten(),
            value=Vol_rec[sample].flatten(),
            isomin=-0.8,
            isomax=0.8,
            opacity=0.1,
            surface_count=30,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Cylinder Spectral Decomposition (B^2 x I)",
        template="plotly_dark",
        height=800,
        scene=dict(aspectratio=dict(x=1, y=1, z=2)),
        scene2=dict(aspectratio=dict(x=1, y=1, z=2)),
    )

    output_file = "cylinder_decomposition.html"
    fig.write_html(output_file)
