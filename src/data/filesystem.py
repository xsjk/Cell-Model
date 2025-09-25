import ftplib
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Self

import numpy as np

from .utils.file import io_to_numpy


class _FTP(ftplib.FTP_TLS):
    trust_server_pasv_ipv4_address = True  # for proxy


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
