import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def transform(func, L, K, x_dim=1000):
    """
    Compute the Discrete Sine Transform (DST) spectral coefficients of a function defined on [0, L].
    Implements a discretized version of the continuous sine transform integral.

    Parameters
    ----------
    func : callable
        A function of one variable (x) defined on the interval [0, L].
        Should satisfy Dirichlet boundary conditions f(0) = f(L) = 0.
    L : float
        The length of the interval.
    K : int
        The number of sine modes to compute.
    x_dim : int, optional
        Number of spatial sample points for numerical integration.

    Returns
    -------
    coeffs : np.ndarray
        A 1D array of shape (K,) containing the spectral coefficients B_k.
    """
    ks, Xs = np.ogrid[1 : K + 1, 0 : L : x_dim * 1j]  # (K, 1), (1, x_dim)
    dx = L / (x_dim - 1)
    Fx = func(Xs.ravel()) * dx  # (x_dim,)

    S = np.sin(ks * np.pi * Xs / L)  # (K, x_dim)
    coeffs = (2 / L) * np.einsum("x,kx->k", Fx, S)  # (K,)
    return coeffs


def forward_transform_fast(func, L, K, x_dim=1000):
    """
    Compute DST spectral coefficients using FFT for speed (O(N log N)).
    Equivalent to transform but faster for large x_dim.

    Parameters
    ----------
    func : callable
        Function of x.
    L : float
        Interval length.
    K : int
        Number of modes.
    x_dim : int
        Number of spatial sample points.

    Returns
    -------
    coeffs : np.ndarray
        Spectral coefficients.
    """
    Xs = np.linspace(0, L, x_dim)
    dx = L / (x_dim - 1)

    inner = func(Xs[1:-1]) * dx
    extended = np.concatenate([[0], inner, [0], -inner[::-1]])
    spectrum = np.fft.fft(extended)
    coeffs = -spectrum[1 : K + 1].imag / L

    return coeffs


def inverse_transform_fast(coeffs, L, N=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct function from DST coeffs using IFFT.
    Faster than inverse_transform for large N.

    Parameters
    ----------
    coeffs : np.ndarray
        Spectral coefficients B_k.
    L : float
        Interval length.
    N : int
        Number of output points.

    Returns
    -------
    X : np.ndarray
        grid points.
    f : np.ndarray
        reconstructed values.
    """
    K = len(coeffs)
    dx = L / (N - 1)
    M = 2 * (N - 1)
    assert K < N - 1

    spectrum = np.zeros(M, dtype=np.complex128)
    spectrum[1 : K + 1] = -1j * L * coeffs[:K]
    spectrum[M - K :] = 1j * L * coeffs[:K][::-1]

    f = np.fft.ifft(spectrum)[:N].real / dx
    Xs = np.linspace(0, L, N)
    return Xs, f


def inverse_transform(coeffs, L, N=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the function from its DST spectral coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        1D array of shape (K,) containing the spectral coefficients B_k.
    L : float
        The length of the interval.
    N : int, optional
        The number of points in the reconstruction grid.

    Returns
    -------
    X : np.ndarray
        The x-coordinates of the grid.
    f : np.ndarray
        The reconstructed function values on the grid.
    """
    K = len(coeffs)
    ks, Xs = np.mgrid[1 : K + 1, 0 : L : N * 1j]  # (K, N)

    S = np.sin(ks * np.pi * Xs / L)  # (K, N)
    f = np.einsum("k,kx->x", coeffs, S)  # (N,)

    return Xs[0, :], f


if __name__ == "__main__":

    def example_func(x):
        # A function satisfying f(0)=f(L)=0
        # Combination of gaussians dampened at borders
        mu1, sigma1 = 30, 10
        mu2, sigma2 = 70, 15

        g1 = np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        g2 = -0.5 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))

        # Enforce boundary conditions cleanly (though gaussians are near 0 approx)
        # Multiply by window function sin(pi*x/L) to ensure exact zero at boundaries
        boundary_window = np.sin(np.pi * x / 100)
        return (g1 + g2) * boundary_window

    LENGTH = 100.0
    K_MAX = 20  # Number of sine modes

    # Transform
    coeffs = transform(example_func, L=LENGTH, K=K_MAX)
    coeffs_fast = forward_transform_fast(example_func, L=LENGTH, K=K_MAX)
    assert np.allclose(coeffs, coeffs_fast)

    # Reconstruct
    X_rec, Y_rec = inverse_transform(coeffs, L=LENGTH, N=500)
    X_rec_fast, Y_rec_fast = inverse_transform_fast(coeffs, L=LENGTH, N=500)
    assert np.allclose(Y_rec, Y_rec_fast)

    # Get ground truth
    Y_true = example_func(X_rec)

    # Visualization
    fig = make_subplots(rows=1, cols=1, subplot_titles=(f"Original vs Reconstructed (K={K_MAX}, L={LENGTH})",))
    fig.add_trace(go.Scatter(x=X_rec, y=Y_true, mode="lines", name="Original Function"))
    fig.add_trace(go.Scatter(x=X_rec, y=Y_rec, mode="lines", name="Reconstructed", line=dict(dash="dash")))
    fig.update_layout(
        title_text="Discrete Sine Transform (DST) Spectral Decomposition",
        template="plotly_dark",
        height=600,
        xaxis_title="x",
        yaxis_title="f(x)",
    )
    fig.write_html("dst_decomposition.html")
