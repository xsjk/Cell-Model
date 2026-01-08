import glob
import os
from abc import ABC
from enum import Enum
from io import BytesIO
from typing import TypedDict

import numpy as np

from . import ndmask
from .utils import media, memory
from ..config import load_dataset_config as load_config


class LoadingMode(Enum):
    FULL_MEMORY = "full_memory"
    COMPRESSED_CACHE = "compressed_cache"
    ON_DEMAND = "on_demand"


class ShapeInfo(TypedDict):
    image_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    image_dtype: str
    mask_dtype: str
    num_samples: int


class SizeInfo(TypedDict):
    image_cache: int
    mask_cache: int
    compressed_cache: int
    total: int


class BaseDataset(ABC):
    mask_ndims: int
    img_ndims: int
    dataset_name: str
    channels: int

    def __init__(self, loading_mode: LoadingMode = LoadingMode.ON_DEMAND, clip: bool = True, config_path: str | None = None):
        self.loading_mode = loading_mode
        self.clipped = clip

        config = load_config(config_path) if config_path is not None else load_config()
        paths = config["Paths"]
        dataset_paths = paths["datasets"][self.dataset_name]

        self.img_dir: str = os.path.join(paths["compressed_dir"], dataset_paths["images"])
        self.mask_dir: str = os.path.join(paths["compressed_dir"], dataset_paths["masks"])
        self.file_list: list[str] = self._scan_files()

        self.img_cache: dict[str, np.ndarray] = {}
        self.mask_cache: dict[str, np.ndarray] = {}
        self.bounds_cache: dict[str, np.ndarray] = {}
        self.compressed_cache: dict[str, tuple[bytes, bytes]] = {}

        match loading_mode:
            case LoadingMode.FULL_MEMORY:
                self._load_all_data()
            case LoadingMode.COMPRESSED_CACHE:
                self._load_all_compressed()
            case LoadingMode.ON_DEMAND:
                pass

    def _scan_files(self) -> list[str]:
        assert os.path.exists(self.img_dir) and os.path.exists(self.mask_dir)
        return sorted(prefix for f in glob.glob(os.path.join(self.img_dir, "*.mp4")) if os.path.exists(os.path.join(self.mask_dir, f"{(prefix := os.path.splitext(os.path.basename(f))[0])}.npz")))

    def _decode_media(self, data: bytes) -> np.ndarray:
        return media.decode_media(data)

    def _decode_mask(self, data: bytes) -> tuple[np.ndarray, np.ndarray]:
        with BytesIO(data) as f:
            m, extra = ndmask.load(f)
            bounds = extra["bounds"]
            return m, bounds

    def _load_image_from_disk(self, prefix: str) -> np.ndarray:
        return media.load_media(os.path.join(self.img_dir, f"{prefix}.mp4"))

    def _load_mask_from_disk(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        with open(os.path.join(self.mask_dir, f"{prefix}.npz"), "rb") as f:
            return self._decode_mask(f.read())

    def _load_compressed_from_disk(self, prefix: str) -> tuple[bytes, bytes]:
        img_path = os.path.join(self.img_dir, f"{prefix}.mp4")
        mask_path = os.path.join(self.mask_dir, f"{prefix}.npz")

        with open(img_path, "rb") as f1, open(mask_path, "rb") as f2:
            return f1.read(), f2.read()

    def _load_all_data(self):
        for prefix in self.file_list:
            self.img_cache[prefix] = self._load_image_from_disk(prefix)
            self.mask_cache[prefix], self.bounds_cache[prefix] = self._load_mask_from_disk(prefix)

    def _load_all_compressed(self):
        for prefix in self.file_list:
            if prefix not in self.compressed_cache:
                self.compressed_cache[prefix] = self._load_compressed_from_disk(prefix)

    def _get_data(self, prefix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match self.loading_mode:
            case LoadingMode.FULL_MEMORY:
                return self.img_cache[prefix], self.mask_cache[prefix], self.bounds_cache[prefix]
            case LoadingMode.COMPRESSED_CACHE:
                img_data, mask_data = self.compressed_cache[prefix]
                mask, bounds = self._decode_mask(mask_data)
                return self._decode_media(img_data), mask, bounds
            case LoadingMode.ON_DEMAND:
                mask, bounds = self._load_mask_from_disk(prefix)
                return self._load_image_from_disk(prefix), mask, bounds

    def _reshape_data(self, img_data: np.ndarray, mask_data: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert img_data.ndim == 3  # since it is stored as video
        assert mask_data.ndim == self.mask_ndims >= 3  # the loaded mask data should have correct dimensions
        assert bounds.shape == (3, 2)

        # Create slices for the mask based on bounds (last 3 dims)
        slices = [slice(None)] * (self.mask_ndims - 3) + [slice(int(s), int(e)) for s, e in bounds]

        # Calculate clipped mask shape by slicing (view)
        # Note: We use tuple(slices) for indexing
        mask_clip_shape = mask_data[tuple(slices)].shape

        if self.img_ndims == self.mask_ndims:
            img_clip = img_data.reshape(mask_clip_shape)
        elif self.img_ndims == self.mask_ndims + 1:
            img_clip = img_data.reshape((-1,) + mask_clip_shape)
            assert self.channels == img_clip.shape[0]
        else:
            raise ValueError(f"Unsupported combination of img ndims and mask ndims: {self.img_ndims}, {self.mask_ndims}")

        if self.clipped:
            return img_clip, mask_data[tuple(slices)]

        # Reconstruct full-size image and mask
        if self.img_ndims == self.mask_ndims:
            img_full = np.zeros(mask_data.shape, dtype=img_data.dtype)
            img_full[*slices] = img_clip
        else:
            img_full = np.zeros((self.channels,) + mask_data.shape, dtype=img_data.dtype)
            img_full[:, *slices] = img_clip

        return img_full, mask_data

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int | str) -> tuple[np.ndarray, np.ndarray]:
        prefix = self.file_list[index] if isinstance(index, int) else index
        img, mask, bounds = self._get_data(prefix)
        return self._reshape_data(img, mask, bounds)

    def clear_cache(self):
        match self.loading_mode:
            case LoadingMode.FULL_MEMORY:
                self.img_cache.clear()
                self.mask_cache.clear()
                self.bounds_cache.clear()
            case LoadingMode.COMPRESSED_CACHE:
                self.compressed_cache.clear()
            case LoadingMode.ON_DEMAND:
                pass

    def get_memory_usage(self) -> SizeInfo:
        sizes = SizeInfo(
            image_cache=memory.get_memory_size(self.img_cache),
            mask_cache=memory.get_memory_size(self.mask_cache) + memory.get_memory_size(self.bounds_cache),
            compressed_cache=memory.get_memory_size(self.compressed_cache),
            total=0,
        )
        sizes["total"] = sum(sizes.values())  # type: ignore
        return SizeInfo(**sizes)

    @property
    def shape_info(self) -> ShapeInfo:
        sample_img, sample_mask = self[0]
        return ShapeInfo(
            image_shape=sample_img.shape,
            mask_shape=sample_mask.shape,
            image_dtype=str(sample_img.dtype),
            mask_dtype=str(sample_mask.dtype),
            num_samples=len(self.file_list),
        )


class WFM(BaseDataset):
    mask_ndims = 5  # (C, T, Z, Y, X)
    img_ndims = 5  # (C, T, Z, Y, X)
    dataset_name = "WFM"
    channels = 2


class SIM(BaseDataset):
    mask_ndims = 3  # (Z, Y, X)
    img_ndims = 4  # (C, Z, Y, X)
    dataset_name = "SIM"
    channels = 3


class SXT(BaseDataset):
    mask_ndims = 3  # (Z, Y, X)
    img_ndims = 3  # (Z, Y, X)
    dataset_name = "SXT"
    channels = 1


class CryoET(BaseDataset):
    mask_ndims = 3  # (Z, Y, X)
    img_ndims = 3  # (Z, Y, X)
    dataset_name = "Cryo-ET"
    channels = 1
