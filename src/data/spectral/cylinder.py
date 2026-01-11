import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.special as sp
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator, interp1d

from .disk import get_alphas, get_bessel_basis


def transform(func, R, L, M, K_r, K_z, N_r=60, N_theta=200, N_z=60):
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

    Returns
    -------
    coeffs : np.ndarray
        Shape (2M+1, K_r, K_z).
    """
    alphas = get_alphas(M, K_r)  # (2M+1, K_r)

    dz = L / N_z
    dr = R / (N_r - 1)
    dt = 2 * np.pi / N_theta

    Rs, Thetas = np.ogrid[0 : R : N_r * 1j, 0 : 2 * np.pi : 2 * np.pi / N_theta]  # (N_r, 1), (1, N_t)
    Zs = np.linspace(dz / 2, L - dz / 2, N_z).reshape(1, 1, N_z)  # (1, 1, N_z)

    Frdrdtdz = func(Rs[:, :, None], Thetas[:, :, None], Zs) * Rs[:, :, None] * dr * dt * dz  # (N_r, N_t, N_z)

    m = np.arange(-M, M + 1)  # (2M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)

    S = np.sin(l[:, None] * np.pi * Zs.ravel()[None, :] / L)  # (K_z, N_z)
    E = np.exp(-1j * m[:, None] * Thetas.ravel()[None, :])  # (2M+1, N_t)
    J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)

    coeffs = np.einsum("rtz, mt, lz, mkr -> mkl", Frdrdtdz, E, S, J, optimize=True)  # (2M+1, K_r, K_z)
    coeffs /= (np.pi * L * (R**2) / 2.0) * (sp.jv(m[:, None, None] + 1, alphas[:, :, None]) ** 2)  # (2M+1, K_r, 1)
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

    r = np.linspace(0, R, N_r)  # (N_r,)
    m = np.arange(-M, M + 1)  # (2M+1,)
    l = np.arange(1, K_z + 1)  # (K_z,)
    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)

    # Create 3D grids for return
    Xs_3d = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))
    Ys_3d = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))
    Zs_3d = np.broadcast_to(z[:, None, None], (N_z, N_xy, N_xy))

    S = np.sin(l[:, None] * np.pi * z[None, :] / L)  # (K_z, N_z)
    J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    E = np.exp(1j * m[:, None, None] * Ts[None, :, :])  # (2M+1, N_xy, N_xy)

    C = np.einsum("mkl, lz, mkr -> mrz", coeffs, S, J)  # (2M+1, N_r, N_z)
    C = interp1d(r, C, axis=1, bounds_error=False, fill_value=0)(Rs)  # (2M+1, N_xy, N_xy, N_z)
    f = np.einsum("mxyz, mxy -> zxy", C, E)  # (N_z, N_xy, N_xy)

    return Xs_3d, Ys_3d, Zs_3d, np.real(f)


def transform_fast(func, R, L, M, K_r, K_z, N_r=60, N_theta=200, N_z=60):
    Rs = np.linspace(0, R, N_r)  # (N_r,)
    Thetas = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)  # (N_theta,)
    dz = L / N_z
    Zs = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)
    m = np.arange(-M, M + 1)  # (2M+1,)

    f = func(Rs[:, None, None], Thetas[None, :, None], Zs[None, None, :])  # (N_r, N_theta, N_z)

    dt = 2 * np.pi / N_theta
    dr = R / (N_r - 1)

    # FFT along Theta (axis 1)
    F: np.ndarray = scipy.fft.fft(f, axis=1)  # type: ignore
    F = F[:, m, :] * dt  # (N_r, 2M+1, N_z)

    # DST along Z (axis 2)
    F = scipy.fft.dst(F, axis=2)  # type: ignore
    F = F[:, :, :K_z] * dz  # (N_r, 2M+1, K_z)

    # Radial projection
    alphas = get_alphas(M, K_r)
    J = get_bessel_basis(M, K_r, N_r)
    coeffs = np.einsum("rml, mkr, r -> mkl", F, J, Rs * dr, optimize=True)  # (2M+1, K_r, K_z)
    coeffs /= (np.pi * L * (R**2)) * (sp.jv(m[:, None, None] + 1, alphas[:, :, None]) ** 2)

    return coeffs


def inverse_transform_fast(coeffs, R, L, N_xy=100, N_z=100, N_r=None, N_theta=None):
    M, K_r, K_z = (coeffs.shape[0] - 1) // 2, coeffs.shape[1], coeffs.shape[2]
    N_r = N_r or int(np.hypot(N_xy, N_xy))
    N_theta = N_theta or int(2 * np.pi * N_xy)

    dz = L / N_z
    m = np.arange(-M, M + 1)  # (2M+1,)
    t = np.linspace(0, 2 * np.pi, N_theta + 1)  # (N_theta+1,)
    r = np.linspace(0, R, N_r)  # (N_r,)
    z = np.linspace(dz / 2, L - dz / 2, N_z)  # (N_z,)

    # IDST along Z (axis 2): K_z -> N_z
    C = scipy.fft.idst(coeffs * N_z, axis=2, n=N_z)  # type: ignore

    # Inverse Bessel transform along R (axis 1): K_r -> N_r
    J = get_bessel_basis(M, K_r, N_r)  # (2M+1, K_r, N_r)
    C: np.ndarray = np.einsum("mkz, mkr -> mrz", C, J)  # (2M+1, N_r, N_z)

    # IFFT along Theta (axis 0): 2M+1 -> N_theta
    F = np.zeros((N_theta, N_r, N_z), dtype=np.complex128)  # (N_theta, N_r, N_z)
    F[m] = C
    f: np.ndarray = scipy.fft.ifft(F, axis=0)  # type: ignore
    f = np.real(f) * N_theta  # (N_theta, N_r, N_z)
    f = np.pad(f, ((0, 1), (0, 0), (0, 0)), mode="wrap")  # (N_theta+1, N_r, N_z)

    # Interpolate Cylindrical -> Cartesian
    Ys, Xs = np.mgrid[-R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (N_xy, N_xy)
    Xs = np.broadcast_to(Xs[None, ...], (N_z, N_xy, N_xy))
    Ys = np.broadcast_to(Ys[None, ...], (N_z, N_xy, N_xy))
    Zs = np.broadcast_to(z[:, None, None], (N_z, N_xy, N_xy))
    Ts = np.broadcast_to(Ts[None, ...], (N_z, N_xy, N_xy))
    Rs = np.broadcast_to(Rs[None, ...], (N_z, N_xy, N_xy))
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

    R_CYL = 100
    L_CYL = 200
    M_MAX = 10
    K_R = 20
    K_Z = 20
    N_R = 50
    N_THETA = 20
    N_Z = 50

    # Transform
    coeffs = transform(example_func, R=R_CYL, L=L_CYL, M=M_MAX, K_r=K_R, K_z=K_Z, N_r=N_R, N_theta=N_THETA, N_z=N_Z)
    coeffs_fast = transform_fast(example_func, R=R_CYL, L=L_CYL, M=M_MAX, K_r=K_R, K_z=K_Z, N_r=N_R, N_theta=N_THETA, N_z=N_Z)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    Xs, Ys, Zs, Vol_rec = inverse_transform(coeffs, R=R_CYL, L=L_CYL, N_xy=N_R * 2, N_z=N_Z)
    Xs_f, Ys_f, Zs_f, Vol_rec_fast = inverse_transform_fast(coeffs, R=R_CYL, L=L_CYL, N_xy=N_R * 2, N_z=N_Z)
    print(f"Inverse Transform max diff: {np.max(np.abs(Vol_rec - Vol_rec_fast)):.6e}")

    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)

    # Get ground truth
    Vol_true = example_func(Rs, Ts, Zs)
    Vol_true[Rs > R_CYL] = 0

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
