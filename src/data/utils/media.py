import json
import subprocess

import numpy as np


def decode_media(data: bytes) -> np.ndarray:
    # Get media info
    res = subprocess.run(
        ["ffprobe", "-show_entries", "stream=width,height,pix_fmt,nb_frames", "-of", "json", "-"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", errors="ignore"))
    s = json.loads(res.stdout.decode("utf-8"))["streams"][0]
    width, height, pix_fmt = int(s["width"]), int(s["height"]), s["pix_fmt"]
    is_video = "nb_frames" in s
    is_gray = "gray" in pix_fmt

    # Decode with ffmpeg, uint8 gray/rgb formats only
    res = subprocess.run(
        ["ffmpeg", "-i", "pipe:", "-f", "rawvideo", "-pix_fmt", "gray" if is_gray else "rgb24", "pipe:"],
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", errors="ignore"))

    arr = np.frombuffer(res.stdout, np.uint8)
    return arr.reshape(((-1,) if is_video else ()) + (height, width) + (() if is_gray else (3,)))


def load_media(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_media(f.read())


def to_mp4(frames: np.ndarray, output_path: str, fps: int = 30, crf: int = 30, preset: str = "veryslow", silent: bool = True) -> None:
    # video: (T, H, W), uint8, grey
    assert frames.ndim == 3, "frames must be a 3D array"
    assert frames.dtype == np.uint8, "frames must be of type uint8"
    T, H, W = frames.shape
    assert H % 2 == 0 and W % 2 == 0, "Height and Width must be even numbers"

    # fmt: off
    proc = subprocess.run(
        [
            "ffmpeg", 
            "-y", 
            "-f", "rawvideo", 
            "-pix_fmt", "gray", 
            "-s", f"{W}x{H}", 
            "-r", str(fps), 
            "-i", "pipe:", 
            "-movflags", "+faststart", 
            "-c:v", "libx265", 
            "-pix_fmt", "gray",
            "-crf", str(crf), 
            "-preset", preset, 
            "-f", "mp4", 
            output_path,
        ],
        input=frames.tobytes(),
        stdout=subprocess.DEVNULL if silent else None,
        stderr=subprocess.DEVNULL if silent else None,
    )
    # fmt: on
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
