"""
Utility modules for data processing.

This package contains specialized utility modules:
- media: Media processing (decode/encode video, images)
- image: Image processing and manipulation
- memory: Memory usage analysis
- file: File format decoding
"""

# Re-export commonly used functions for convenience
from .media import decode_media, load_media, to_mp4
from .image import show_img, auto_scale, auto_clip, get_bounding_box, u16_to_u8, keep_largest_connected_component, parametric_standarize
from .memory import get_memory_size
from .file import io_to_numpy

__all__ = [
    # Media processing
    "decode_media",
    "load_media",
    "to_mp4",
    # Image processing
    "show_img",
    "auto_scale",
    "auto_clip",
    "get_bounding_box",
    "u16_to_u8",
    "keep_largest_connected_component",
    "parametric_standarize",
    # Memory analysis
    "get_memory_size",
    # File processing
    "io_to_numpy",
]
