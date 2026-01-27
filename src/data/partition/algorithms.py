import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import equalize


def partition_edt(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    gamma_cell: float = 1.0,
    gamma_nuc: float = 1.0,
    equalization: bool = False,
) -> np.ndarray:
    """
    Generate partition using Euclidean Distance Transform (EDT).

    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell (3D). 1 inside, 0 outside.
    nucleus_mask : np.ndarray
        Binary mask of the nucleus (3D). 1 inside, 0 outside.
    gamma_cell : float
        Gamma factor for cell EDT.
    gamma_nuc : float
        Gamma factor for nucleus EDT.
    equalization : bool
        Whether to apply histogram equalization.

    Returns
    -------
    partition : np.ndarray
        Scalar field p in [0, 1] defined on cytoplasm, 0 in nucleus, 1 outside cell.
    """
    d_cell = ndimage.distance_transform_edt(cell_mask) ** gamma_cell  # type: ignore[assignment]
    d_nuc = ndimage.distance_transform_edt(~nucleus_mask) ** gamma_nuc  # type: ignore[assignment]
    p = d_nuc / (d_cell + d_nuc)

    if equalization:
        cytoplasm_mask = cell_mask & (~nucleus_mask)
        p = equalize.equalize_scalar_field(p, mask=cytoplasm_mask)

    return p


def partition_laplace(
    cell_mask: np.ndarray,
    nucleus_mask: np.ndarray,
    max_iter: int = 100000,
    tol: float = 1e-5,
    init: np.ndarray | None = None,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    equalization: bool = False,
) -> np.ndarray:
    r"""
    Generate partition using Laplace equation.

    Solves:
        \nabla^2 u = 0 in C \ K
        u = 0 on \partial K (nucleus)
        u = 1 on \partial C (cell)

    Parameters
    ----------
    cell_mask : np.ndarray
        Binary mask of the cell (3D).
    nucleus_mask : np.ndarray
        Binary mask of the nucleus (3D).
    max_iter : int
        Maximum iterations for the solver (Jacobi).
    tol : float
        Convergence tolerance.
    init : np.ndarray | None
        Initial guess for u.
    device : str
        Compute device.
    equalization : bool
        Whether to apply histogram equalization.

    Returns
    -------
    partition : np.ndarray
        Scalar field p in [0, 1] defined on cytoplasm, 0 in nucleus, 1 outside cell.
    """

    C = torch.as_tensor(cell_mask, dtype=torch.bool, device=device)
    K = torch.as_tensor(nucleus_mask, dtype=torch.bool, device=device)

    if init is not None:
        u = torch.as_tensor(init, dtype=torch.float32, device=device)
    else:
        u = torch.zeros_like(C, dtype=torch.float32, device=device)
        u[~C] = 1.0
        u[C & ~K] = 0.5

    # Kernel for Jacobi iteration (average of 6 neighbors)
    k = torch.tensor(
        [
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.float32,
        device=device,
    )
    k = k[None, None, ...] / 6.0  # (1,1,3,3,3)

    for i in range(max_iter):
        u_ = u.clone()  # (Z,Y,X)
        u = F.conv3d(u[None, None, ...], k, padding=1)[0, 0]  # (Z,Y,X)
        u[K] = 0.0
        u[~C] = 1.0
        if i % 100 == 0:
            diff = (u - u_).abs().max()
            if diff < tol:
                break
    else:
        logging.warning("partition_laplace: did not converge within max_iter")

    u[K] = 0.0
    u[~C] = 1.0

    p = u.cpu().numpy()  # (Z,Y,X)

    if equalization:
        cytoplasm_mask = cell_mask & (~nucleus_mask)
        p = equalize.equalize_scalar_field(p, mask=cytoplasm_mask)

    return p
