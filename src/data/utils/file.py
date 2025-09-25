import warnings
from typing import IO

import numpy as np

from . import media


def io_to_numpy(io: IO[bytes], extension: str) -> np.ndarray:
    assert io.readable()
    assert io.tell() == 0
    match extension:
        case ".npy":
            return np.load(io)

        case ".tif" | ".tiff":
            import tifffile

            return tifffile.imread(io)

        case ".czi":
            from czifile import CziFile

            with CziFile(io) as czi:
                return czi.asarray()

        case ".mrc":
            from mrcfile.mrcfile import MrcInterpreter

            with warnings.catch_warnings(category=RuntimeWarning, action="ignore"):
                it = MrcInterpreter(iostream=io, permissive=True)
                assert (data := it.data) is not None
                return data

        case ".png" | ".jpg" | ".jpeg" | ".mp4":
            return media.decode_media(io.read())

        case _:
            raise ValueError(f"Unsupported file type: {extension}")
