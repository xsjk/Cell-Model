import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
from tqdm import tqdm

from . import ndmask
from .filesystem import LocalFS
from .utils import auto_clip, keep_largest_connected_component, parametric_standarize, to_mp4, u16_to_u8
from .utils.config import load as load_config

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile.mrcinterpreter")


config = load_config()
paths_config = config["Paths"]
processing_config = config.get("Processing", {})

# Setup directories
download_dir = paths_config["original_dir"]
compressed_dir = paths_config["compressed_dir"]
os.makedirs(compressed_dir, exist_ok=True)

fs = LocalFS()


def _check_skip_conditions(required_files, output_files, img_file_name):
    # Check if all required files exist
    if not all(os.path.exists(f) for f in required_files):
        return f"Skipped: {img_file_name} (required files {', '.join(required_files)} not found)"

    # Check if already compressed
    if all(os.path.exists(f) for f in output_files):
        return f"Skipped: {img_file_name} (already compressed)"

    return None  # No skip condition met


def _process_files_parallel(files, process_func, desc, *args):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_func, file_name, *args): file_name for file_name in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()


def get_dataset_dirs(dataset_name):
    dataset_paths = paths_config["datasets"][dataset_name]

    download_img_dir = os.path.join(download_dir, dataset_paths["images"])
    download_mask_dir = os.path.join(download_dir, dataset_paths["masks"])
    compressed_img_dir = os.path.join(compressed_dir, dataset_paths["images"])
    compressed_mask_dir = os.path.join(compressed_dir, dataset_paths["masks"])

    # Create compressed directories
    os.makedirs(compressed_img_dir, exist_ok=True)
    os.makedirs(compressed_mask_dir, exist_ok=True)

    return download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir


def _compress_wfm_single(img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
    file_prefix = img_file_name[:-4]  # remove .tif
    mask_file_name = f"{file_prefix}_PM_NE_mask.npy"
    compressed_img_name = f"{file_prefix}.mp4"
    compressed_mask_name = f"{file_prefix}.npz"

    img_file_path = os.path.join(download_img_dir, img_file_name)
    mask_file_path = os.path.join(download_mask_dir, mask_file_name)
    compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
    compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

    # Check common skip conditions
    if skip_reason := _check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
        if "required files not found" in skip_reason:
            print(f"Mask file not found for {img_file_name}, expected {mask_file_name}")
        return skip_reason

    try:
        # Load image and mask
        img = fs.read_numpy(img_file_path)
        mask = fs.read_numpy(mask_file_path)
        assert img.dtype == np.uint8
        assert mask.dtype == np.uint8
        assert img.shape == mask.shape

        # Clip and normalize
        clip, mask_clip, _ = auto_clip(img, mask, axes=[-2, -1])
        assert clip.shape == mask_clip.shape

        # Compress and save
        to_mp4(clip.transpose(2, 0, 1, 3, 4).reshape(-1, *clip.shape[-2:]), compressed_img_path)
        ndmask.save(compressed_mask_path, mask_clip)

        return f"Completed: {img_file_name}"
    except Exception as e:
        return f"Error processing {img_file_name}: {str(e)}"


def _compress_sim_single(img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir, best_scales, best_gammas):
    """Process a single SIM image file for compression."""
    file_prefix = img_file_name[:-10]  # remove _Actin.tif
    img_file_names = [f"{file_prefix}_Actin.tif", f"{file_prefix}_ISG.tif", f"{file_prefix}_N.tif"]
    mask_file_names = [f"{file_prefix}_N.mrc", f"{file_prefix}_PM.mrc"]
    compressed_img_name = f"{file_prefix}.mp4"
    compressed_mask_name = f"{file_prefix}.npz"

    img_file_paths = [os.path.join(download_img_dir, name) for name in img_file_names]
    mask_file_paths = [os.path.join(download_mask_dir, name) for name in mask_file_names]
    compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
    compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

    # Check common skip conditions
    if skip_reason := _check_skip_conditions(img_file_paths + mask_file_paths, [compressed_img_path, compressed_mask_path], img_file_name):
        return skip_reason

    try:
        # Load image and mask
        img = np.array([fs.read_numpy(p) for p in img_file_paths])
        mask = np.sum([fs.read_numpy(p) << i for i, p in enumerate(mask_file_paths)], axis=0, dtype=np.uint8)
        assert isinstance(mask, np.ndarray)
        assert img.dtype == np.uint16
        assert mask.dtype == np.uint8
        assert img.shape[1:] == mask.shape

        # Clip and normalize
        clip, mask_clip, _ = auto_clip(img, mask, axes=[-3, -2, -1])
        assert clip.shape[1:] == mask_clip.shape
        assert img.dtype == np.uint16

        # Gamma correction
        clip_adjust = np.zeros_like(clip, dtype=np.uint8)
        for i in range(3):
            clip_adjust[i] = u16_to_u8(clip[i], gamma=best_gammas[i], scale=best_scales[i])

        # Compress and save
        to_mp4(clip_adjust.reshape(-1, *clip.shape[-2:]), compressed_img_path)
        ndmask.save(compressed_mask_path, mask_clip)

        return f"Completed: {img_file_name}"
    except Exception as e:
        return f"Error processing {img_file_name}: {str(e)}"


def _compress_sxt_single(img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
    if m := re.match(r"Stevens_pancreatic_I[Nn][Ss]_1E_(?:25mM_|25-10_30min_|3_)?([\d_]*)_pre_rec\.mrc", img_file_name):
        file_prefix = m.group(1)  # 785_7, ... , 931_9_10
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

    # Check common skip conditions
    if skip_reason := _check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
        return skip_reason

    try:
        # Load image and mask
        img = fs.read_numpy(img_file_path)
        mask = fs.read_numpy(mask_file_path)
        assert img.dtype == np.float32
        assert mask.dtype == np.uint8
        assert img.shape == mask.shape

        mask = keep_largest_connected_component(mask)

        # Clip and normalize
        img_clip, mask_clip, _ = auto_clip(img, mask, axes=[-3, -2, -1])
        assert img_clip.shape == mask_clip.shape
        assert img_clip.dtype == np.float32
        assert mask_clip.dtype == np.uint8

        # Standardize
        img_clip_adjust = parametric_standarize(img_clip, mask=mask_clip)

        # Compress and save
        to_mp4(img_clip_adjust.reshape(-1, *img_clip.shape[-2:]), compressed_img_path)
        ndmask.save(compressed_mask_path, mask_clip)

        return f"Completed: {img_file_name}"
    except Exception as e:
        return f"Error processing {img_file_name}: {str(e)}"


def _compress_cryo_et_single(img_file_name, download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir):
    if m := re.match(r"(\d+_[\d\.]+_\d+)_denoise[d]?\.mrc", img_file_name):
        file_prefix = m.group(1)  # 20220219_5_2, 20220918_30_001, 20210323_2.8_010, ...
    else:
        return f"Skipped: {img_file_name} (unrecognized file name)"

    mask_file_name = f"{file_prefix}_ISG_Filled.mrc"
    compressed_img_name = f"{file_prefix}.mp4"
    compressed_mask_name = f"{file_prefix}.npz"

    img_file_path = os.path.join(download_img_dir, img_file_name)
    mask_file_path = os.path.join(download_mask_dir, mask_file_name)
    compressed_img_path = os.path.join(compressed_img_dir, compressed_img_name)
    compressed_mask_path = os.path.join(compressed_mask_dir, compressed_mask_name)

    # Check common skip conditions
    if skip_reason := _check_skip_conditions([mask_file_path], [compressed_img_path, compressed_mask_path], img_file_name):
        return skip_reason

    try:
        # Load image and mask
        img = fs.read_numpy(img_file_path)
        mask = fs.read_numpy(mask_file_path)
        assert img.dtype == np.float32
        assert mask.dtype == np.int8
        assert img.shape == mask.shape
        mask = mask.view(dtype=np.uint8)

        # Standardize
        img_adjust = parametric_standarize(img, mask=mask)
        assert img_adjust.dtype == np.uint8
        assert img_adjust.shape == img.shape

        # Compress and save
        to_mp4(img_adjust.reshape(-1, *img.shape[-2:]), compressed_img_path)
        ndmask.save(compressed_mask_path, mask)

        return f"Completed: {img_file_name}"
    except Exception as e:
        return f"Error processing {img_file_name}: {str(e)}"


def _compress_dataset(dataset_name: str, file_filter: Callable[[str], bool], process_func: Callable, extra_args: list = []):
    download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir = get_dataset_dirs(dataset_name)
    img_files = [p for p in os.listdir(download_img_dir) if file_filter(p)]
    args = [download_img_dir, download_mask_dir, compressed_img_dir, compressed_mask_dir] + extra_args
    _process_files_parallel(img_files, process_func, f"Compressing {dataset_name}", *args)


def compress_wfm():
    _compress_dataset("WFM", lambda p: p.endswith(".tif"), _compress_wfm_single)


def compress_sim():
    # Get processing parameters from config
    sim_config = processing_config["SIM"]
    _compress_dataset("SIM", lambda p: p.endswith("_Actin.tif"), _compress_sim_single, [sim_config["scales"], sim_config["gammas"]])


def compress_sxt():
    _compress_dataset("SXT", lambda p: True, _compress_sxt_single)


def compress_cryo_et():
    _compress_dataset("Cryo-ET", lambda p: p.endswith(".mrc"), _compress_cryo_et_single)


compress_wfm()
compress_sim()
compress_sxt()
compress_cryo_et()
