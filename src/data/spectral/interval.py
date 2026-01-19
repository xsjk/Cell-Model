import numpy as np
import scipy.fft
import torch


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


def transform_fast(func, L, K, N=1000) -> np.ndarray:
    x = samples(N)  # (N，)
    f = func(x * L)  # (N,)
    return scipy.fft.dst(f, overwrite_x=True)[:K] / N  # type: ignore


def inverse_transform_fast(coeffs, L, N=256) -> tuple[np.ndarray, np.ndarray]:
    x = samples(N)  # (N，)
    f = scipy.fft.idst(coeffs * N, n=N, overwrite_x=True)
    return x * L, f  # type: ignore[return-value]


def transform_torch(data, K):
    assert data.ndim == 1
    device, dtype = data.device, data.dtype
    N = data.shape[0]
    x = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N
    k = torch.arange(1, K + 1, device=device, dtype=dtype)
    S = torch.sin(k[:, None] * torch.pi * x[None, :])  # (K, N)
    coeffs = (2 / N) * torch.einsum("x, kx -> k", data, S)  # (K,)
    return coeffs


def inverse_transform_torch(coeffs, N):
    assert coeffs.ndim == 1
    device, dtype = coeffs.device, coeffs.dtype
    K = coeffs.shape[0]
    x = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N
    k = torch.arange(1, K + 1, device=device, dtype=dtype)
    S = torch.sin(k[:, None] * torch.pi * x[None, :])  # (K, N)
    f = torch.einsum("k, kx -> x", coeffs, S)  # (N,)
    return f  # type: ignore[return-value]


def fit(data, K, activation=None, reg=None, init_coeffs=None, lr=5e-2, criterion=None, epochs=1000, log_interval=None, device=None):
    """
    Compute spectral coefficients by fitting reconstruction to sampled values, supporting custom post-processing.
    """
    assert data.ndim == 1
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device=device)
    else:
        data = data.detach().to(device=device)

    N = data.shape[0]  # infer N from input
    dtype = data.dtype

    if init_coeffs is None:
        coeffs = transform_torch(data, K).requires_grad_(True)
    else:
        coeffs = torch.tensor(init_coeffs, dtype=dtype, device=device, requires_grad=True)

    criterion = criterion or torch.nn.functional.mse_loss
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs) if epochs > 50 else None

    for i in range(epochs):
        optimizer.zero_grad()
        f = inverse_transform_torch(coeffs, N)
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
        f = inverse_transform_torch(coeffs, N)
        if activation:
            f = activation(f)

    return coeffs.detach().cpu().numpy(), f.detach().cpu().numpy()


if __name__ == "__main__":
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    c_dir = transform(observed_func, L, K, N=N)
    c_fast = transform_fast(observed_func, L, K, N=N)
    Xs, f_rec = inverse_transform(c_dir, L, N=N)
    Xs, f_fast = inverse_transform_fast(c_dir, L, N=N)
    f_torch = inverse_transform_torch(torch.from_numpy(c_dir).to(device), N).cpu().numpy()
    np.testing.assert_allclose(f_torch, f_fast, atol=1e-14)

    f_obs = observed_func(Xs)  # (N,)

    print("Fitting...")
    activation = lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX)  # noqa: E731
    c_fit, f_fit_torch = fit(f_obs, K, activation=activation, init_coeffs=c_dir, log_interval=100)

    Xs, f_fit_raw = inverse_transform(c_fit, L, N=N)
    f_fit = np.clip(f_fit_raw, CLIP_MIN, CLIP_MAX)
    np.testing.assert_allclose(f_fit_torch, f_fit, atol=1e-14)

    f_lat = latent_func(Xs)

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
