import glob
import os
from abc import ABC
from enum import Enum
from io import BytesIO
from typing import TypedDict

import numpy as np

from . import ndmask, utils
from .config import load_config


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
    image_cache_mb: float
    mask_cache_mb: float
    compressed_cache_mb: float
    total_mb: float


class BaseDataset(ABC):
    mask_ndims: int
    img_ndims: int
    dataset_name: str

    def __init__(self, loading_mode: LoadingMode = LoadingMode.ON_DEMAND):
        self.loading_mode = loading_mode

        config = load_config()
        paths = config["Paths"]
        dataset_paths = paths["datasets"][self.dataset_name]

        self.img_dir: str = os.path.join(paths["compressed_dir"], dataset_paths["images"])
        self.mask_dir: str = os.path.join(paths["compressed_dir"], dataset_paths["masks"])
        self.file_list: list[str] = self._scan_files()

        self.img_cache: dict[str, np.ndarray] = {}
        self.mask_cache: dict[str, np.ndarray] = {}
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

    @staticmethod
    def _decode_media(data: bytes) -> np.ndarray:
        return utils.decode_media(data)

    @staticmethod
    def _decode_mask(data: bytes) -> np.ndarray:
        with BytesIO(data) as f:
            return ndmask.load(f)

    def _load_image_from_disk(self, prefix: str) -> np.ndarray:
        return utils.load_media(os.path.join(self.img_dir, f"{prefix}.mp4"))

    def _load_mask_from_disk(self, prefix: str) -> np.ndarray:
        return ndmask.load(os.path.join(self.mask_dir, f"{prefix}.npz"))

    def _load_compressed_from_disk(self, prefix: str) -> tuple[bytes, bytes]:
        img_path = os.path.join(self.img_dir, f"{prefix}.mp4")
        mask_path = os.path.join(self.mask_dir, f"{prefix}.npz")

        with open(img_path, "rb") as f1, open(mask_path, "rb") as f2:
            return f1.read(), f2.read()

    def _load_all_data(self):
        for prefix in self.file_list:
            self.img_cache[prefix] = self._load_image_from_disk(prefix)
            self.mask_cache[prefix] = self._load_mask_from_disk(prefix)

    def _load_all_compressed(self):
        for prefix in self.file_list:
            if prefix not in self.compressed_cache:
                self.compressed_cache[prefix] = self._load_compressed_from_disk(prefix)

    def _get_data(self, prefix) -> tuple[np.ndarray, np.ndarray]:
        match self.loading_mode:
            case LoadingMode.FULL_MEMORY:
                return self.img_cache[prefix], self.mask_cache[prefix]
            case LoadingMode.COMPRESSED_CACHE:
                img_data, mask_data = self.compressed_cache[prefix]
                return self._decode_media(img_data), self._decode_mask(mask_data)
            case LoadingMode.ON_DEMAND:
                return self._load_image_from_disk(prefix), self._load_mask_from_disk(prefix)

    def _reshape_data(self, img_data: np.ndarray, mask_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert img_data.ndim == 3  # since it is stored as video
        assert mask_data.ndim == self.mask_ndims  # the loaded mask data should have correct dimensions

        # Check shape compatibility
        if self.img_ndims > 3:
            assert img_data.shape[-2:] == mask_data.shape[-2:]
        elif self.img_ndims == 3:
            assert img_data.shape[-3:] == mask_data.shape[-3:]
        else:
            raise ValueError(f"img ndims should larger than or equal to mask ndims, got {self.img_ndims} < {self.mask_ndims}")

        # Reshape the image data to target dimensions and match the mask shape
        if self.img_ndims == self.mask_ndims:
            return img_data.reshape(mask_data.shape), mask_data
        elif self.img_ndims == self.mask_ndims + 1:
            return img_data.reshape((-1,) + mask_data.shape), mask_data
        else:
            raise ValueError(f"Unsupported combination of img ndims and mask ndims: {self.img_ndims}, {self.mask_ndims}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int | str) -> tuple[np.ndarray, np.ndarray]:
        prefix = self.file_list[index] if isinstance(index, int) else index
        return self._reshape_data(*self._get_data(prefix))

    def clear_cache(self):
        match self.loading_mode:
            case LoadingMode.FULL_MEMORY:
                self.img_cache.clear()
                self.mask_cache.clear()
            case LoadingMode.COMPRESSED_CACHE:
                self.compressed_cache.clear()
            case LoadingMode.ON_DEMAND:
                pass

    def get_memory_usage(self) -> SizeInfo:
        mb = 1024 * 1024
        sizes = SizeInfo(
            image_cache_mb=utils.get_memory_size(self.img_cache) / mb,
            mask_cache_mb=utils.get_memory_size(self.mask_cache) / mb,
            compressed_cache_mb=utils.get_memory_size(self.compressed_cache) / mb,
            total_mb=0.0,
        )
        sizes["total_mb"] = sum(sizes.values())  # type: ignore
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
    mask_ndims = 5
    img_ndims = 5
    dataset_name = "WFM"


class SIM(BaseDataset):
    mask_ndims = 3
    img_ndims = 4
    dataset_name = "SIM"


class SXT(BaseDataset):
    mask_ndims = 3
    img_ndims = 3
    dataset_name = "SXT"


class CryoET(BaseDataset):
    mask_ndims = 3
    img_ndims = 3
    dataset_name = "Cryo-ET"
