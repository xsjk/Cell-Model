import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import exposure


def show_img(im: np.ndarray, ax=None) -> None:
    if ax is None:
        ax = plt.gca()
    assert im.ndim in (2, 3)
    if im.ndim == 3:
        assert im.shape[0] == 3
        im = im.transpose(1, 2, 0)
        ax.imshow(im)
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def auto_scale(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        assert mask.dtype == np.bool_
        if arr.shape != mask.shape:
            assert arr.shape[-mask.ndim :] == mask.shape
            channel_shape, shape = arr.shape[: -mask.ndim], arr.shape[-mask.ndim :]
            channel_count = np.prod(channel_shape)
            arr = arr.reshape((channel_count,) + shape)
            return np.array([auto_scale(arr[i], mask=mask) for i in range(channel_count)]).reshape(channel_shape + shape)

        assert arr.shape == mask.shape
        masked = arr[mask]
        a_min, a_max = masked.min(), masked.max()
    else:
        a_min, a_max = arr.min(), arr.max()
    assert a_min < a_max
    return np.clip((arr - a_min) / (a_max - a_min), 0, 1)


def get_bounding_box(mask, axes=[-3, -2, -1], even=True) -> tuple[slice, ...]:
    assert np.any(mask), "Mask is empty"
    nonzero = np.nonzero(mask)
    res = tuple()
    for axis in axes:
        min_idx = int(np.min(nonzero[axis]))
        max_idx = int(np.max(nonzero[axis]))
        if even and (max_idx - min_idx) & 1:
            min_idx += 1
        res += (slice(min_idx, max_idx),)
    return res


def auto_clip(arr: np.ndarray, mask: np.ndarray, scale: bool = True, even: bool = True, axes: list[int] = [-3, -2, -1]) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...]]:
    arr_dtype = arr.dtype
    assert mask.dtype == np.bool_ or np.issubdtype(mask.dtype, np.unsignedinteger)
    assert arr.shape[-mask.ndim :] == mask.shape
    bound = get_bounding_box(mask > 0, axes=axes, even=even)
    # access the axes of arr
    slices = [slice(None)] * mask.ndim
    for ax in axes:
        slices[ax] = slice(bound[ax].start, bound[ax].stop)
    arr = arr[..., *slices]
    mask = mask[*slices]
    if scale:
        arr = auto_scale(arr, mask=mask > 0)
    if np.issubdtype(arr_dtype, np.unsignedinteger):
        arr = (arr * np.iinfo(arr_dtype).max).astype(arr_dtype)
    elif np.issubdtype(arr_dtype, np.floating):
        assert arr.dtype == arr_dtype
    else:
        raise TypeError("Unsupported array dtype")
    return arr, mask, bound


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    assert image.dtype == np.uint8
    inv_gamma = 1 / gamma
    table = (np.linspace(0, 1, 256) ** inv_gamma * 255).astype(image.dtype)
    return cv2.LUT(image, table)


def calculate_entropy(pixels):
    prob, _ = exposure.histogram(pixels, normalize=True)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def information_retention_ratio(entropy_original, entropy_transformed):
    retention_ratio = entropy_transformed / entropy_original
    loss_percentage = 1 - retention_ratio
    return loss_percentage


def u16_to_u8(image: np.ndarray, gamma: float, scale: float) -> np.ndarray:
    assert image.dtype == np.uint16
    return (np.clip(image / 65535 * scale, 0, 1) ** (1 / gamma) * 255).astype(np.uint8)


def number_of_components(mask: np.ndarray, connectivity: int = 1) -> int:
    labeled, ncomponents = ndimage.label(mask > 0)  # type: ignore
    return ncomponents


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled_mask, num_labels = ndimage.label(mask)  # type: ignore
    assert num_labels > 0
    component_sizes = np.bincount(labeled_mask.ravel())
    max_label = component_sizes[1:].argmax() + 1
    largest_component = labeled_mask == max_label
    mask = mask.copy()
    mask[~largest_component] = 0
    return mask


def parametric_standarize(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    percentiles = np.percentile(image[mask > 0] if mask is not None else image, [40, 50, 60])
    img_center = percentiles[1]
    img_coef = 1 / (np.diff(percentiles).mean() / 0.1)
    img_adjust = ((1 / (1 + np.exp(-4 * img_coef * (image - img_center)))).astype(image.dtype) * 255).astype(np.uint8)
    return img_adjust


def print_err(diff, factor=1):
    print("Median Absolute Error: ", np.median(np.abs(diff)) / factor)
    print("Mean Absolute Error: ", np.mean(np.abs(diff)) / factor)
    print("Root Mean Squared Error: ", np.sqrt(np.mean(diff**2)) / factor)
