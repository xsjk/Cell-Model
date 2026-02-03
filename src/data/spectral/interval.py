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


def evaluate_basis_torch(coeffs, x):
    assert coeffs.ndim == 1
    device, dtype = coeffs.device, coeffs.dtype
    K = coeffs.shape[0]
    k = torch.arange(1, K + 1, device=device, dtype=dtype)
    S = torch.sin(k[:, None] * torch.pi * x[None, :])
    f = torch.einsum("k, kx -> x", coeffs, S)
    return f


def inverse_transform_torch(coeffs, N, bounds=(0, 1)):
    assert coeffs.ndim == 1
    device, dtype = coeffs.device, coeffs.dtype
    a, b = bounds[0], bounds[1]
    x = a + (b - a) * ((torch.arange(N, device=device, dtype=dtype) + 0.5) / N)
    return evaluate_basis_torch(coeffs, x)


def fit(
    data,
    K,
    activation=None,
    reg=None,
    init_coeffs=None,
    lr=5e-2,
    criterion=None,
    epochs=1000,
    log_interval=None,
    device=None,
    bounds=(0, 1),
):
    """
    Compute spectral coefficients by fitting reconstruction to sampled values, supporting custom post-processing.

    Returns
    -------
    coeffs : np.ndarray
        Fitted spectral coefficients.
    f_fit : np.ndarray
        Reconstructed function values on the observation grid.
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
        f = inverse_transform_torch(coeffs, N, bounds=bounds)

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
        f = inverse_transform_torch(coeffs, N, bounds=bounds)
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

    # Data Setup
    TRUE_A, TRUE_B = 0, 0.8
    x = (TRUE_A + (TRUE_B - TRUE_A) * samples(N)) * L
    f_obs = observed_func(x)
    f_lat = latent_func(x)
    x_full = samples(N) * L
    f_lat_full = latent_func(x_full)
    f_obs_full = observed_func(x_full)

    print("=" * 58)
    print(f"Interval Spec. Decomp. (L={L}, K={K})")
    print(f"Observation interval: [{TRUE_A:.2f}, {TRUE_B:.2f}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transform and Reconstruction
    ## Direct Transform (assumes full interval [0, 1])
    c_dir = scipy.fft.dst(f_obs, overwrite_x=False)[:K] / N  # type: ignore[assignment]
    _, f_rec = inverse_transform(c_dir, L, N=N)

    ## Fitted Recon (Fixed bounds [0, 1] with clipping activation)
    print("Fitting (Fixed Bounds)...")
    activation = lambda x: torch.clamp(x, CLIP_MIN, CLIP_MAX)  # noqa: E731
    c_fit, f_fit = fit(f_obs, K, activation=activation, log_interval=200)

    ## Fitted Recon (Manual bounds [TRUE_A, TRUE_B] with clipping activation)
    print("Fitting (Manual Bounds)...")
    c_fit_man, f_fit_man = fit(f_obs, K, activation=activation, bounds=(TRUE_A, TRUE_B), log_interval=200)

    def get_full_recon(coeffs: np.ndarray, bounds=(0, 1)) -> np.ndarray:
        with torch.no_grad():
            a, b = bounds[0], bounds[1]
            u = (samples(N) - TRUE_A) / (TRUE_B - TRUE_A)
            u = torch.from_numpy(a + (b - a) * u).to(device=device, dtype=torch.float32)
            c_torch = torch.from_numpy(coeffs).to(device=device, dtype=torch.float32)
            return evaluate_basis_torch(c_torch, u).cpu().numpy()

    f_fit_full = get_full_recon(c_fit)
    f_fit_man_full = get_full_recon(c_fit_man, bounds=(TRUE_A, TRUE_B))

    # Visualization
    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "Observed (Truncated X&Y)",
            "Direct (Fixed [0,1])",
            "Fitted (Fixed [0,1])",
            "Fitted (Manual [a,b])",
            "",  # Empty slot
            "Direct Error (Trunc.)",
            "Fixed Fit Error (Trunc.)",
            "Manual Fit Error (Trunc.)",
        ),
        vertical_spacing=0.15,
    )

    def add_line(r, c, x, y, name, color=None, dash=None, showlegend=True):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash),
                showlegend=showlegend,
            ),
            row=r,
            col=c,
        )

    for col in range(1, 5):
        add_line(1, col, x_full, f_lat_full, "Latent Full", "gray", "dash", showlegend=(col == 1))
    add_line(1, 1, x, f_obs, "Observed", "cyan")
    add_line(1, 2, x, f_rec, "Direct Slice", "orange")
    add_line(1, 3, x_full, f_fit_full, "Fixed Fit Extrap.", "magenta", "dot")
    add_line(1, 3, x, f_fit, "Fixed Fit Slice", "magenta", showlegend=False)
    add_line(1, 4, x_full, f_fit_man_full, "Manual Fit Extrap.", "yellow", "dot")
    add_line(1, 4, x, f_fit_man, "Manual Fit Slice", "yellow", showlegend=False)
    add_line(2, 2, x, np.abs(f_rec - f_obs), "Err Direct", "red")
    add_line(2, 3, x, np.abs(f_fit - f_obs), "Err Fixed", "red")
    add_line(2, 4, x, np.abs(f_fit_man - f_obs), "Err Manual", "red")
    v_min, v_max = f_lat_full.min(), f_lat_full.max()
    max_err = max(np.abs(f_rec - f_obs).max(), np.abs(f_fit_man - f_obs).max())

    fig.update_xaxes(range=[0, L])
    fig.update_yaxes(range=[v_min, v_max], row=1)
    fig.update_yaxes(range=[0, max_err], row=2)
    fig.update_layout(
        title=f"Adaptive Interval Benchmark (L={L}, K={K})",
        template="plotly_dark",
    )
    fig.write_html("interval_decomposition.html", include_plotlyjs="cdn")
