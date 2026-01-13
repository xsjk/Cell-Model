import numpy as np
import plotly.graph_objects as go
import scipy.fft
from plotly.subplots import make_subplots


def samples(N):
    return np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)


def transform(func, L, K, N=1000):
    """
    Use the sine basis to compute the spectral coefficients of a real-valued function defined on an interval [0, L].
    The function should satisfy Dirichlet boundary conditions f(0) = f(L) = 0.

        **Domain**:
            Interval = I = [0, L]

        **Basis**:
            Sine basis: sin(k * pi * x / L)
            for k = 1, 2, ..., K.

    Parameters
    ----------
    func : callable
        A function of one variable (x) defined on the interval [0, L].
        Should satisfy Dirichlet boundary conditions f(0) = f(L) = 0.
    L : float
        The length of the interval.
    K : int
        The number of sine modes to compute.
    N : int, optional
        Number of spatial sample points for numerical integration.

    Returns
    -------
    coeffs : np.ndarray
        A 1D array of shape (K,) containing the spectral coefficients B_k.
    """
    x = samples(N)  # (N，)
    k = np.arange(1, K + 1)  # (K,)
    f = func(x * L)  # (N,)
    S = np.sin(k[:, None] * np.pi * x[None, :])  # (K, N)
    coeffs = 2 / N * np.einsum("x,kx->k", f, S)  # (K,)
    return coeffs


def inverse_transform(coeffs, L, N=256) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct the function from its sine spectral coefficients.

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
    x = samples(N)  # (N，)
    k = np.arange(1, K + 1)  # (K,)
    S = np.sin(k[:, None] * np.pi * x[None, :])  # (K, N)
    f = np.einsum("k,kx->x", coeffs, S)  # (N,)
    return x * L, f


def transform_fast(func, L, K, N=1000):
    """
    Fast computation of the sine spectral coefficients using the Discrete Sine Transform (DST).
    The function should satisfy Dirichlet boundary conditions f(0) = f(L) = 0.

    See the `transform` function for interface details.
    """
    x = samples(N)  # (N，)
    f = func(x * L)  # (N,)
    return scipy.fft.dst(f, overwrite_x=True)[:K] / N  # type: ignore


def inverse_transform_fast(coeffs, L, N=256):
    """
    Fast reconstruction of the function from its sine spectral coefficients using the Inverse Discrete Sine Transform (IDST).

    See the `inverse_transform` function for interface details.
    """
    x = samples(N)  # (N，)
    f = scipy.fft.idst(coeffs * N, n=N, overwrite_x=True)
    return x * L, f


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

    # Display errors
    print("\nError vs Ground Truth:")
    print(f"MAE: {np.max(np.abs(Y_true - Y_rec)):.6e}")
    print(f"RMSE: {np.sqrt(np.mean((Y_true - Y_rec) ** 2)):.6e}")

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
