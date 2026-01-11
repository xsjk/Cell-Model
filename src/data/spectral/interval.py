import numpy as np
import plotly.graph_objects as go
import scipy.fft
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
    dx = L / x_dim
    Xs = np.linspace(dx / 2, L - dx / 2, x_dim).reshape(1, x_dim)  # (1, x_dim)
    ks = np.arange(1, K + 1).reshape(K, 1)  # (K, 1)

    Fx = func(Xs.ravel()) * dx  # (x_dim,)

    S = np.sin(ks * np.pi * Xs / L)  # (K, x_dim)
    coeffs = (2 / L) * np.einsum("x,kx->k", Fx, S)  # (K,)
    return coeffs


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
    dx = L / N
    Xs = np.linspace(dx / 2, L - dx / 2, N).reshape(1, N)  # (1, N)
    ks = np.arange(1, K + 1).reshape(K, 1)  # (K, 1)

    S = np.sin(ks * np.pi * Xs / L)  # (K, N)
    f = np.einsum("k,kx->x", coeffs, S)  # (N,)

    return Xs.ravel(), f


def transform_fast(func, L, K, x_dim=1000):
    dx = L / x_dim
    Xs = np.linspace(dx / 2, L - dx / 2, x_dim)
    return scipy.fft.dst(func(Xs))[:K] / x_dim  # type: ignore


def inverse_transform_fast(coeffs, L, N=256):
    dx = L / N
    f = scipy.fft.idst(coeffs * N, n=N)
    return np.linspace(dx / 2, L - dx / 2, N), f


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
    coeffs_fast = transform_fast(example_func, L=LENGTH, K=K_MAX)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    X_rec, Y_rec = inverse_transform(coeffs, L=LENGTH, N=500)
    X_rec_fast, Y_rec_fast = inverse_transform_fast(coeffs, L=LENGTH, N=500)
    print(f"Inverse Transform max diff: {np.max(np.abs(Y_rec - Y_rec_fast)):.6e}")

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
