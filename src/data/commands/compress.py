import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
from tqdm import tqdm

from .. import ndmask
from ..filesystem import LocalFS
from ..utils import auto_clip, keep_largest_connected_component, parametric_standarize, to_mp4, u16_to_u8
from ...config import load_dataset_config as load_config

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile.mrcinterpreter")


class DataCompressor:
    def __init__(self, config_path: str | None = None):
        config = load_config() if config_path is None else load_config(config_path)
        self.paths_config = config["Paths"]
        self.processing_config = config.get("Processing", {})

        self.download_dir = self.paths_config["original_dir"]
        self.compressed_dir = self.paths_config["compressed_dir"]
        os.makedirs(self.compressed_dir, exist_ok=True)

        self.fs = LocalFS()

    def _check_skip_conditions(self, required_files, output_files, img_file_name):
        if not all(os.path.exists(f) for f in required_files):
            return f"Skipped: {img_file_name} (required files {', '.join(required_files)} not found)"
        if all(os.path.exists(f) for f in output_files):
            return f"Skipped: {img_file_name} (already compressed)"
        return None

    def _process_files_parallel(self, files, process_func, desc, *args):
        max_workers = min(os.cpu_count() or 1, 32)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_func, file_name, *args): file_name for file_name in files}
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                res = future.result()
                if "Error" in res:
                    print(res)

    def get_dataset_dirs(self, dataset_name):
        dataset_paths = self.paths_config["datasets"][dataset_name]

        download_img_dir = os.path.join(self.download_dir, dataset_paths["images"])
        download_mask_dir = os.path.join(self.download_dir, dataset_paths["masks"])
        compressed_img_dir = os.path.join(self.compressed_dir, dataset_paths["images"])
        compressed_mask_dir = os.path.join(self.compressed_dir, dataset_paths["masks"])

        os.makedirs(compressed_img_dir, exist_ok=True)
        os.makedirs(compressed_mask_dir, exist_ok=True)

        return download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir

    def _compress_wfm_single(self, img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
        file_prefix = img_file_name[:-4]
        mask_file_name = f"{file_prefix}_PM_NE_mask.npy"
        compressed_img_name = f"{file_prefix}.mp4"
        compressed_mask_name = f"{file_prefix}.npz"

        img_file_path = os.path.join(download_img_dir, img_file_name)
        mask_file_path = os.path.join(download_mask_dir, mask_file_name)
        compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
        compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

        if skip_reason := self._check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
            if "required files not found" in skip_reason:
                print(f"Mask file not found for {img_file_name}, expected {mask_file_name}")
            return skip_reason

        try:
            img = self.fs.read_numpy(img_file_path)
            mask = self.fs.read_numpy(mask_file_path)
            assert img.dtype == np.uint8 and mask.dtype == np.uint8 and img.shape == mask.shape

            clip, mask_clip, _ = auto_clip(img, mask, axes=[-2, -1])
            assert clip.shape == mask_clip.shape

            to_mp4(clip.transpose(2, 0, 1, 3, 4).reshape(-1, *clip.shape[-2:]), compressed_img_path)
            ndmask.save(compressed_mask_path, mask_clip)

            return f"Completed: {img_file_name}"
        except Exception as e:
            return f"Error processing {img_file_name}: {str(e)}"

    def _compress_sim_single(self, img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir, best_scales, best_gammas):
        file_prefix = img_file_name[:-10]
        img_file_names = [f"{file_prefix}_Actin.tif", f"{file_prefix}_ISG.tif", f"{file_prefix}_N.tif"]
        mask_file_names = [f"{file_prefix}_N.mrc", f"{file_prefix}_PM.mrc"]
        compressed_img_name = f"{file_prefix}.mp4"
        compressed_mask_name = f"{file_prefix}.npz"

        img_file_paths = [os.path.join(download_img_dir, name) for name in img_file_names]
        mask_file_paths = [os.path.join(download_mask_dir, name) for name in mask_file_names]
        compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
        compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

        if skip_reason := self._check_skip_conditions(img_file_paths + mask_file_paths, [compressed_img_path, compressed_mask_path], img_file_name):
            return skip_reason

        try:
            img = np.array([self.fs.read_numpy(p) for p in img_file_paths])
            mask = np.sum([self.fs.read_numpy(p) << i for i, p in enumerate(mask_file_paths)], axis=0, dtype=np.uint8)
            assert isinstance(img, np.ndarray) and isinstance(mask, np.ndarray)
            assert img.dtype == np.uint16 and mask.dtype == np.uint8 and img.shape[1:] == mask.shape

            clip, mask_clip, _ = auto_clip(img, mask, axes=[-3, -2, -1])
            assert clip.shape[1:] == mask_clip.shape

            clip_adjust = np.zeros_like(clip, dtype=np.uint8)
            for i in range(3):
                clip_adjust[i] = u16_to_u8(clip[i], gamma=best_gammas[i], scale=best_scales[i])

            to_mp4(clip_adjust.reshape(-1, *clip.shape[-2:]), compressed_img_path)
            ndmask.save(compressed_mask_path, mask_clip)

            return f"Completed: {img_file_name}"
        except Exception as e:
            return f"Error processing {img_file_name}: {str(e)}"

    def _compress_sxt_single(self, img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
        if m := re.match(r"Stevens_pancreatic_I[Nn][Ss]_1E_(?:25mM_|25-10_30min_|3_)?([\d_]*)_pre_rec\.mrc", img_file_name):
            file_prefix = m.group(1)
        else:
            return f"Skipped: {img_file_name} (unrecognized file name)"

        part = file_prefix.split("_")
        assert len(part) in (2, 3)
        if len(part) == 3:
            part[1:] = sorted(part[1:], key=int)
            file_prefix = f"{part[0]}_{part[1]}_{part[0]}_{part[2]}"

        mask_file_name = f"{file_prefix}_merged_manual_mask.tiff"
        compressed_img_name = f"{file_prefix}.mp4"
        compressed_mask_name = f"{file_prefix}.npz"

        img_file_path = os.path.join(download_img_dir, img_file_name)
        mask_file_path = os.path.join(download_mask_dir, mask_file_name)
        compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
        compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

        if skip_reason := self._check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
            return skip_reason

        try:
            img = self.fs.read_numpy(img_file_path)
            mask = self.fs.read_numpy(mask_file_path)
            assert img.dtype == np.float32 and mask.dtype == np.uint8 and img.shape == mask.shape

            mask = keep_largest_connected_component(mask)
            img_clip, mask_clip, _ = auto_clip(img, mask, axes=[-3, -2, -1])
            assert img_clip.shape == mask_clip.shape

            img_clip_adjust = parametric_standarize(img_clip, mask=mask_clip)
            to_mp4(img_clip_adjust.reshape(-1, *img_clip.shape[-2:]), compressed_img_path)
            ndmask.save(compressed_mask_path, mask_clip)

            return f"Completed: {img_file_name}"
        except Exception as e:
            return f"Error processing {img_file_name}: {str(e)}"

    def _compress_cryo_et_single(self, img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
        if m := re.match(r"(\d+_[\d\.]+_\d+)_denoise[d]?\.mrc", img_file_name):
            file_prefix = m.group(1)
        else:
            return f"Skipped: {img_file_name} (unrecognized file name)"

        mask_file_name = f"{file_prefix}_ISG_Filled.mrc"
        compressed_img_name = f"{file_prefix}.mp4"
        compressed_mask_name = f"{file_prefix}.npz"

        img_file_path = os.path.join(download_img_dir, img_file_name)
        mask_file_path = os.path.join(download_mask_dir, mask_file_name)
        compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
        compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

        if skip_reason := self._check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
            return skip_reason

        try:
            img = self.fs.read_numpy(img_file_path)
            mask = self.fs.read_numpy(mask_file_path)
            assert img.dtype == np.float32 and mask.dtype == np.int8 and img.shape == mask.shape
            mask = mask.view(dtype=np.uint8)

            img_adjust = parametric_standarize(img, mask=mask)
            assert img_adjust.dtype == np.uint8 and img_adjust.shape == img.shape

            to_mp4(img_adjust.reshape(-1, *img.shape[-2:]), compressed_img_path)
            ndmask.save(compressed_mask_path, mask)

            return f"Completed: {img_file_name}"
        except Exception as e:
            return f"Error processing {img_file_name}: {str(e)}"

    def _compress_dataset(self, dataset_name: str, file_filter: Callable[[str], bool], process_func: Callable, extra_args: list = []):
        download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir = self.get_dataset_dirs(dataset_name)
        img_files = [p for p in os.listdir(download_img_dir) if file_filter(p)]
        args = [download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir] + extra_args
        self._process_files_parallel(img_files, process_func, f"Compressing {dataset_name}", *args)

    def run(self):
        sim_config = self.processing_config["SIM"]
        self._compress_dataset("WFM", lambda p: p.endswith(".tif"), self._compress_wfm_single)
        self._compress_dataset("SIM", lambda p: p.endswith("_Actin.tif"), self._compress_sim_single, [sim_config["scales"], sim_config["gammas"]])
        self._compress_dataset("SXT", lambda _: True, self._compress_sxt_single)
        self._compress_dataset("Cryo-ET", lambda p: p.endswith(".mrc"), self._compress_cryo_et_single)


def main(config_path: str | None = None):
    DataCompressor(config_path).run()
