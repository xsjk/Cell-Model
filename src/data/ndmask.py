from typing import Any, Mapping, NamedTuple, Self

import numpy as np

type DenseMask = np.ndarray
type Mask = SparseMask | DenseMask


class SparseMask(NamedTuple):
    axis: int
    shape: tuple[int, ...]
    coords: tuple[np.ndarray, ...]
    data: np.ndarray
    first: Mask

    def flatten(self) -> dict:
        """
        Convert SparseMask to a flat dictionary for saving.

        Returns
        -------
        dict
            Flattened representation with keys like 'axis', 'shape', 'coords[0]', 'data', etc.
        """
        result = {"axis": self.axis, "shape": self.shape, "data": self.data}

        for i, coord in enumerate(self.coords):
            result[f"coords[{i}]"] = coord

        if isinstance(self.first, SparseMask):
            for k, v in self.first.flatten().items():
                result[f"first.{k}"] = v
        else:
            result["first"] = self.first

        return result

    @classmethod
    def reconstruct(cls, data: Mapping[str, Any]) -> Self:
        """
        Reconstruct SparseMask from a flattened mapping.

        Parameters
        ----------
        data : Mapping[str, Any]
            Flattened representation as produced by `flatten()`.

        Returns
        -------
        SparseMask
            Reconstructed SparseMask instance.
        """

        return cls(
            int(data["axis"]),
            tuple(map(int, data["shape"])),
            tuple(data[f"coords[{i}]"] for i in range(len([k for k in data if k.startswith("coords")]))),
            data["data"],
            data["first"] if "first" in data else cls.reconstruct({key[6:]: value for key, value in data.items() if key.startswith("first.")}),
        )


def _get_diff(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Compute the difference array along the axis with the least variation.

    Parameters
    ----------
    arr : np.ndarray
        Input array of dtype int8.

    Returns
    -------
    e : np.ndarray
        Difference array with the same shape as `arr`.
    axis : int
        Axis along which the difference was calculated.
    """
    assert np.issubdtype(arr.dtype, np.signedinteger)
    axis = min(range(arr.ndim), key=lambda axis: np.abs(np.diff(arr, axis=axis)).mean())
    e = np.diff(arr, axis=axis, prepend=np.array(0, dtype=arr.dtype))  # assume no overflow
    return e, axis


def _minimize_dtype(arr: np.ndarray) -> np.ndarray:
    """
    Convert the input array to the smallest possible integer dtype
    that can hold its values.

    Parameters
    ----------
    arr : np.ndarray
        Input array with non-negative values.

    Returns
    -------
    arr_minimized : np.ndarray
        Array converted to the smallest suitable integer dtype.
    """
    assert np.issubdtype(arr.dtype, np.integer)
    signed = np.issubdtype(arr.dtype, np.signedinteger)
    if arr.size == 0:
        return arr.astype(np.int8 if signed else np.uint8)
    min_val = arr.min()
    max_val = arr.max()
    for dtype in (np.int8, np.int16, np.int32, np.int64) if signed else (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_val <= np.iinfo(dtype).max and (not signed or min_val >= np.iinfo(dtype).min):
            return arr.astype(dtype)
    raise ValueError("Array contains values too large for any supported dtype.")


def _dense_to_coo(arr: np.ndarray) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """
    Convert a binary mask array to COO (Coordinate) format.

    Parameters
    ----------
    arr : np.ndarray
        Input binary mask array of dtype uint8.

    Returns
    -------
    coords : tuple of np.ndarray
        Tuple containing the coordinates of non-zero elements.
    data : np.ndarray
        Array containing the values of non-zero elements.
    """
    coord = np.nonzero(arr)
    data = arr[*coord]
    return tuple(_minimize_dtype(c) for c in coord), data


def _coo_to_dense(coords: tuple[np.ndarray, ...], data: np.ndarray, shape: tuple[int, ...], out: np.ndarray | None = None) -> np.ndarray:
    """
    Convert COO (Coordinate) format back to a dense binary mask array.

    Parameters
    ----------
    coords : tuple of np.ndarray
        Tuple containing the coordinates of non-zero elements.
    data : np.ndarray
        Array containing the values of non-zero elements.
    shape : tuple of int
        Shape of the output dense array.
    out : np.ndarray, optional
        Preallocated zero array to store the result. If None, a new array is created.

    Returns
    -------
    arr : np.ndarray
        Dense binary mask array of dtype uint8.
    """
    if out is None:
        out = np.zeros(shape, dtype=data.dtype)
    out[*coords] = data
    return out


def _get_sparse_repr(mask: np.ndarray) -> SparseMask:
    """
    Convert a binary mask array to a sparse representation.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask array of dtype uint8.

    Returns
    -------
    axis : int
        Axis along which the difference was calculated.
    shape : tuple of int
        Shape of the original mask array.
    coords : tuple of np.ndarray
        Tuple containing the coordinates of non-zero elements in the difference array.
    data : np.ndarray
        Array containing the values of non-zero elements in the difference array.
    """
    e, axis = _get_diff(mask)
    indices = (slice(None),) * axis + (0,)
    first = e[indices]
    if first.ndim > 1:
        first = _get_sparse_repr(first)
        e[indices] = 0
    coords, data = _dense_to_coo(e)
    return SparseMask(axis, mask.shape, coords, data, first)


def _restore_mask(axis: int, shape: tuple[int, ...], coords: tuple[np.ndarray, ...], data: np.ndarray, first: Mask, out: np.ndarray | None = None) -> np.ndarray:
    """
    Restore the original binary mask array from its sparse representation.

    Parameters
    ----------
    axis : int
        Axis along which the difference was calculated.
    shape : tuple of int
        Shape of the original mask array.
    coords : tuple of np.ndarray
        Tuple containing the coordinates of non-zero elements in the difference array.
    data : np.ndarray
        Array containing the values of non-zero elements in the difference array.
    first : Mask
        The first slice along the specified axis, can be a dense array or another sparse representation.
    out : np.ndarray, optional
        Preallocated array to store the restored mask. If None, a new array is created.

    Returns
    -------
    mask : np.ndarray
        Restored binary mask array of dtype uint8.
    """
    e = np.zeros(shape, dtype=data.dtype) if out is None else out
    _coo_to_dense(coords, data, shape, out=e)

    indices = (slice(None),) * axis + (0,)
    if isinstance(first, tuple):
        _restore_mask(*first, out=e[indices])
    else:
        e[indices] = first
    return np.cumsum(e, axis=axis, dtype=e.dtype, out=e)


def save(filename: str, mask: np.ndarray) -> None:
    """
    Save a binary mask array in a compressed sparse format.

    Parameters
    ----------
    filename : str
        Path to the npz file to save the sparse representation.
    mask : np.ndarray
        Input binary mask array of dtype uint8.
    """
    assert mask.dtype == np.uint8
    spmask = _get_sparse_repr(mask.view(dtype=np.int8))
    np.savez_compressed(filename, allow_pickle=False, **spmask.flatten())


def load(filename: str) -> np.ndarray:
    """
    Load a binary mask array from a compressed sparse format.

    Parameters
    ----------
    filename : str
        Path to the npz file containing the sparse representation.

    Returns
    -------
    mask : np.ndarray
        Restored binary mask array of dtype uint8.
    """
    spmask = SparseMask.reconstruct(np.load(filename, allow_pickle=False))
    mask = _restore_mask(*spmask)
    assert mask.dtype == np.int8
    return mask.view(dtype=np.uint8)
