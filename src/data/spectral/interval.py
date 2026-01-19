import numpy as np
import scipy.fft


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
    x = samples(N)  # (N，)
    f = func(x * L)  # (N,)
    return scipy.fft.dst(f, overwrite_x=True)[:K] / N  # type: ignore


def inverse_transform_fast(coeffs, L, N=256):
    x = samples(N)  # (N，)
    f = scipy.fft.idst(coeffs * N, n=N, overwrite_x=True)
    return x * L, f


def fit(func, L, K, N=1000, activation=None, reg=None, init_coeffs=None, lr=1e-2, criterion=None, epochs=200, verbose=False, device=None):
    """
    Compute spectral coefficients by fitting reconstruction to sampled values, supporting custom post-processing.

    Parameters
    ----------
    func : callable
        A function of one variable (x) defined on the interval [0, L].
    L : float
        The length of the interval.
    K : int
        The number of sine modes to compute.
    N : int, optional
        Number of spatial sample points for fitting.
    activation : callable, optional
        Post-processing function applied to the reconstruction before computing loss.
    reg : callable, optional
        Regularization function on coefficients. Default is no regularization.
    init_coeffs : np.ndarray, optional
        Initial coefficients for optimization. If None, uses direct transform.
    lr : float
        Learning rate.
    criterion : callable, optional
        Loss function (pred, target). Default is MSE.
    epochs : int
        Number of optimization iterations.
    verbose : bool
        Whether to print progress.
    device : str, optional
        Device for computation ('cpu' or 'cuda'). Defaults to 'cuda' if available.

    Returns
    -------
    coeffs : np.ndarray
        Optimized coefficients of shape (K,).

    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Sample on grid
    x = samples(N)  # (N,)
    k = np.arange(1, K + 1)  # (K,)

    # Target values
    f_true = func(x * L)  # (N,)

    S = np.sin(k[:, None] * np.pi * x[None, :])  # (K, N)

    if init_coeffs is None:
        init_coeffs = transform(func, L, K, N)
    coeffs = torch.tensor(init_coeffs, dtype=torch.float32, device=device, requires_grad=True)

    S = torch.from_numpy(S).to(device=device, dtype=torch.float32)
    f_true = torch.from_numpy(f_true).to(device=device, dtype=torch.float32)

    criterion = criterion or torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam([coeffs], lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()

        f = torch.einsum("k, kx -> x", coeffs, S)

        if activation:
            f = activation(f)

        loss = criterion(f, f_true)
        if reg:
            loss = loss + reg(coeffs)

        loss.backward()
        optimizer.step()

        if verbose and i % 10 == 0:
            print(f"Iter {i}: Loss {loss.item():.6e}")

    return coeffs.detach().cpu().numpy()


if __name__ == "__main__":
    import torch
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def latent_func(x):
        mu1, sigma1 = 30, 8
        mu2, sigma2 = 70, 10
        g1 = 1.5 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        g2 = -1.2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
        return (g1 + g2) * soft_mask(x, L=L)

    CLIP_MIN, CLIP_MAX = -0.5, 0.5

    def observed_func(x):
        return np.clip(latent_func(x), CLIP_MIN, CLIP_MAX)

    L = 100.0
    K = 30
    N = 500

    print("=" * 58)
    print(f"Interval Spec. Decomp. (L={L}, K={K})")

    c_dir = transform(observed_func, L, K, N=N)
    c_fit = fit(observed_func, L, K, N=N, activation=lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX), init_coeffs=c_dir, lr=0.05, epochs=400)
    c_fast = transform_fast(observed_func, L, K, N=N)
    Xs, f_rec = inverse_transform(c_dir, L, N=N)
    Xs, f_fast = inverse_transform_fast(c_dir, L, N=N)
    Xs, f_fit_raw = inverse_transform(c_fit, L, N=N)
    f_fit = np.clip(f_fit_raw, CLIP_MIN, CLIP_MAX)
    f_lat = latent_func(Xs)
    f_obs = observed_func(Xs)

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
        vertical_spacing=0.15,
    )

    def add_line(r, c, y, name, color=None, dash=None):
        fig.add_trace(
            go.Scatter(
                x=Xs,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash),
            ),
            row=r,
            col=c,
        )

    add_line(1, 1, f_obs, "Observed", "cyan")
    add_line(1, 2, f_rec, "Direct", "orange")
    add_line(1, 3, f_fit, "Fitted(Clip)", "magenta")
    add_line(1, 3, f_fit_raw, "Fitted(Raw)", "purple", "dot")
    add_line(2, 1, f_lat, "Latent", "green")
    add_line(2, 2, np.abs(f_rec - f_obs), "Err Direct", "red")
    add_line(2, 3, np.abs(f_fit - f_obs), "Err Fitted", "red")

    fs = [f_obs, f_rec, f_fit, f_lat]
    v_min, v_max = min(f.min() for f in fs), max(f.max() for f in fs)
    max_err = np.abs(f_rec - f_obs).max()

    fig.update_layout(
        title="Interval Clipped Reconstruction Benchmark",
        template="plotly_dark",
        yaxis1=dict(range=[v_min, v_max]),
        yaxis2=dict(range=[v_min, v_max]),
        yaxis3=dict(range=[v_min, v_max]),
        yaxis4=dict(range=[v_min, v_max]),
        yaxis5=dict(range=[0, max_err]),
        yaxis6=dict(range=[0, max_err]),
    )
    fig.write_html("interval_decomposition.html", include_plotlyjs="cdn")
