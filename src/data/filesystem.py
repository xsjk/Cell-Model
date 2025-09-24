import ftplib
import io
import json
import os
import subprocess
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import IO, Callable, Self

import numpy as np


class _FTP(ftplib.FTP_TLS):
    trust_server_pasv_ipv4_address = True  # for proxy


def _decode_media_with_ffmpeg(data: bytes) -> np.ndarray:
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


class BaseFS(ABC):
    @abstractmethod
    def read(self, path: str, **kwargs) -> bytes:
        pass

    @abstractmethod
    def read_numpy(self, path: str, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def ls(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def cd(self, path: str) -> None:
        pass

    @abstractmethod
    def pwd(self) -> str:
        pass


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
            return _decode_media_with_ffmpeg(io.read())

        case _:
            raise ValueError(f"Unsupported file type: {extension}")


class LocalFS(BaseFS):
    def read(self, path: str, **kwargs) -> bytes:
        with open(path, "rb", **kwargs) as f:
            return f.read()

    def read_numpy(self, path: str, **kwargs) -> np.ndarray:
        _, ext = os.path.splitext(path)
        with open(path, "rb", **kwargs) as f:
            return io_to_numpy(f, extension=ext)

    def ls(self, path: str) -> list[str]:
        return os.listdir(path)

    def cd(self, path: str) -> None:
        os.chdir(path)

    def pwd(self) -> str:
        return os.getcwd()


def auto_reconnect(method: Callable) -> Callable:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except EOFError:
            print("Connection lost. Reconnecting...")
            self.connect()
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class FTPFS(BaseFS):
    host: str
    port: int
    user: str
    passwd: str

    def __enter__(self) -> Self:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False

    @property
    def ftp(self) -> _FTP:
        if not hasattr(self, "_ftp"):
            self.connect()
        return self._ftp

    @property
    def is_open(self) -> bool:
        return hasattr(self, "_ftp") and self._ftp.sock is not None

    def connect(self) -> Self:
        self._ftp = _FTP()
        self._ftp.connect(self.host, self.port)
        self._ftp.login(self.user, self.passwd)
        self._ftp.prot_p()
        return self

    def disconnect(self) -> None:
        self._ftp.quit()
        del self._ftp

    @auto_reconnect
    def ls(self, path=".") -> list[str]:
        return self.ftp.nlst(path)

    @auto_reconnect
    def cd(self, path) -> None:
        self.ftp.cwd(path)

    @auto_reconnect
    def pwd(self) -> str:
        return self.ftp.pwd()

    @auto_reconnect
    def read(self, filename: str) -> bytes:
        with io.BytesIO() as bio:
            self.ftp.retrbinary(f"RETR {filename}", bio.write)
            return bio.getvalue()

    @auto_reconnect
    def read_numpy(self, filename: str, /, save: str | None = None) -> np.ndarray:
        _, ext = os.path.splitext(filename)
        with io.BytesIO() if save is None else open(save, "wb+") as bio:
            self.ftp.retrbinary(f"RETR {filename}", bio.write)
            bio.seek(0)
            return io_to_numpy(bio, extension=ext)


if __name__ == "__main__":
    import os
    import tomllib

    with open("config.toml", "rb") as f:
        ftp_config = tomllib.load(f)["FTP"]
    ftp = FTPFS(**ftp_config)
