import numpy as np
import scipy.special as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .disk import get_alphas


def transform(func, R, L, M, K_r, K_z, r_dim=60, theta_dim=60, z_dim=60):
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

    Rs, Thetas, Zs = np.ogrid[0 : R : r_dim * 1j, 0 : 2 * np.pi : 2 * np.pi / theta_dim, 0 : L : z_dim * 1j]  # (r,1,1), Thetas: (1,t,1), Zs: (1,1,z)

    dr = R / (r_dim - 1)
    dt = (2 * np.pi) / theta_dim
    dz = L / (z_dim - 1)

    Frdrdtdz = func(Rs, Thetas, Zs) * Rs * dr * dt * dz

    l = np.arange(1, K_z + 1)  # (K_z,)
    m = np.arange(-M, M + 1)  # (2M+1,)
    z = Zs.ravel()  # (z_dim,)
    t = Thetas.ravel()  # (theta_dim,)
    r = Rs.ravel()  # (r_dim,)

    S = np.sin(l[:, None] * np.pi * z[None, :] / L)  # (K_z, z_dim)
    E = np.exp(1j * m[:, None] * t[None, :])  # (2M+1, theta_dim)
    J = sp.jv(m[:, None, None], alphas[:, :, None] * r[None, None, :] / R)  # (2M+1, K_r, r_dim)

    coeffs = np.einsum("rtz, lz, mt, mkr -> mkl", Frdrdtdz, S, E, J, optimize=True)  # (2M+1, K_r, K_z)
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
    M_full, K_r, K_z = coeffs.shape
    M = (M_full - 1) // 2
    alphas = get_alphas(M, K_r)

    Zs, Ys, Xs = np.mgrid[0 : L : N_z * 1j, -R : R : N_xy * 1j, -R : R : N_xy * 1j]  # (N_z, N_xy, N_xy)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)  # (1, N_xy, N_xy)

    # Extract axes
    z_vec = Zs[:, 0, 0]  # (N_z,)
    r_mat = Rs[0, :, :]  # (N_xy, N_xy)
    t_mat = Ts[0, :, :]  # (N_xy, N_xy)

    l = np.arange(1, K_z + 1)
    m = np.arange(-M, M + 1)

    S = np.sin(l[:, None] * np.pi * z_vec[None, :] / L)  # (K_z, N_z)
    J = sp.jv(m[:, None, None, None], alphas[:, :, None, None] * (r_mat[None, None, :, :] / R))  # (2M+1, K_r, N_xy, N_xy)
    E = np.exp(-1j * m[:, None, None] * t_mat[None, :, :])  # (2M+1, N_xy, N_xy)

    f = np.einsum("mkl, lz, mkxy, mxy -> zxy", coeffs, S, J, E, optimize=True)  # (N_z, N_xy, N_xy)
    f = np.real(f)
    f[Rs > R] = 0
    return Xs, Ys, Zs, f


if __name__ == "__main__":

    def example_cylinder_func(r, theta, z):
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

    # Transform
    coeffs = transform(example_cylinder_func, R=R_CYL, L=L_CYL, M=M_MAX, K_r=K_R, K_z=K_Z, r_dim=40, theta_dim=40, z_dim=40)

    # Reconstruct
    Xs, Ys, Zs, Vol_rec = inverse_transform(coeffs, R=R_CYL, L=L_CYL, N_xy=80, N_z=80)
    Rs, Ts = np.hypot(Xs, Ys), np.arctan2(Ys, Xs)

    # Get ground truth
    Vol_true = example_cylinder_func(Rs, Ts, Zs)
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
