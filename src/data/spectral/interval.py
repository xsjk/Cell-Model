import numpy as np
import plotly.graph_objects as go
import scipy.fft
from plotly.subplots import make_subplots


def samples(N):
    return np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)


def soft_mask(x, L=1.0, margin=None):
    """
    Create a soft mask that smoothly transitions from 1 to 0 near the boundaries of the interval [0, L].
    Use cosine tapering over the specified margin.

    Parameters
    ----------
    x : np.ndarray
        Input array of points where the mask is evaluated.
    L : float
        The length of the interval.
    margin : float, optional
        The width over which the mask transitions from 1 to 0 at the boundaries.
    """
    if margin is None:
        margin = L / 2
    assert 0 <= margin <= L / 2, "margin must be less than L/2"
    if margin == 0:
        return ((x >= 0) & (x <= L)).astype(float)
    y1 = 0.5 * (1 - np.cos(np.pi * x / margin))
    y2 = 0.5 * (1 - np.cos(np.pi * (L - x) / margin))
    return (x < margin) * y1 + (x > L - margin) * y2 + ((x >= margin) & (x <= L - margin))


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
        mu1, sigma1 = 30, 10
        mu2, sigma2 = 70, 15
        g1 = np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        g2 = -0.5 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
        return (g1 + g2) * soft_mask(x, L=L)

    L = 100.0  # Length of the interval
    K = 20  # Number of sine modes
    N = 500  # Number of sample points for reconstruction

    # Transform
    coeffs = transform(example_func, L=L, K=K)
    coeffs_fast = transform_fast(example_func, L=L, K=K)
    print(f"Transform max diff: {np.max(np.abs(coeffs - coeffs_fast)):.6e}")

    # Reconstruct
    X_rec, Y_rec = inverse_transform(coeffs, L=L, N=N)
    X_rec_fast, Y_rec_fast = inverse_transform_fast(coeffs, L=L, N=N)
    print(f"Inverse Transform max diff: {np.max(np.abs(Y_rec - Y_rec_fast)):.6e}")

    # Get ground truth
    Y_true = example_func(X_rec)

    # Display errors
    print("\nError vs Ground Truth:")
    print(f"MAE: {np.max(np.abs(Y_true - Y_rec)):.6e}")
    print(f"RMSE: {np.sqrt(np.mean((Y_true - Y_rec) ** 2)):.6e}")

    # Visualization
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(f"Original vs Reconstructed (K={K}, L={L})", "Absolute Error"),
        vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(x=X_rec, y=Y_true, mode="lines", name="Original"), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_rec, y=Y_rec, mode="lines", name="Reconstructed", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_rec, y=np.abs(Y_true - Y_rec), mode="lines", name="Error", line=dict(color="crimson")), row=2, col=1)

    fig.update_layout(
        title_text="Discrete Sine Transform (DST) Spectral Decomposition",
        template="plotly_dark",
    )
    fig.write_html("dst_decomposition.html")
