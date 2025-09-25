import os
import os.path
import re
import socket

import socks
from tqdm import tqdm

from .filesystem import FTPFS
from .utils.config import load as load_config


class DataFetcher:
    def __init__(self, config_path=None):
        config = load_config() if config_path is None else load_config(config_path)
        self.remote_config = config["Remote"]
        self.paths_config = config["Paths"]

        if "proxy" in self.remote_config:
            proxy_url = self.remote_config["proxy"]
            if match := re.match(r"^(socks5|http)://([^:/]+):(\d+)$", proxy_url):
                scheme, host, port = match.groups()
                proxy_type = socks.SOCKS5 if scheme == "socks5" else socks.HTTP
                socks.set_default_proxy(proxy_type, host, int(port))
                socket.socket = socks.socksocket
            else:
                raise ValueError(f"Invalid proxy URL format: {proxy_url}")

        ftp_config = {
            "host": self.remote_config["host"],
            "port": self.remote_config["port"],
            "user": self.remote_config["user"],
            "passwd": self.remote_config["passwd"],
        }
        self.ftp = FTPFS(**ftp_config)

    def fetch_all(self, remote_pattern: str, local_dir: str):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        for path in tqdm(self.ftp.ls(remote_pattern), desc=f"Downloading {remote_pattern} to {local_dir}"):
            remote_dir, filename = os.path.split(path)
            local_path = os.path.join(local_dir, filename)
            if os.path.exists(local_path):
                continue
            self.ftp.read_numpy(path, save=local_path)

    def fetch_dataset(self, dataset_name: str):
        if dataset_name not in self.remote_config["datasets"]:
            raise ValueError(f"Dataset {dataset_name} not found in configuration")

        dataset_remote = self.remote_config["datasets"][dataset_name]
        dataset_paths = self.paths_config["datasets"][dataset_name]

        images_local_dir = os.path.join(self.paths_config["original_dir"], dataset_paths["images"])
        masks_local_dir = os.path.join(self.paths_config["original_dir"], dataset_paths["masks"])

        images_patterns = dataset_remote["images"]
        if isinstance(images_patterns, list):
            for pattern in images_patterns:
                self.fetch_all(pattern, images_local_dir)
        else:
            assert isinstance(images_patterns, str)
            self.fetch_all(images_patterns, images_local_dir)

        self.fetch_all(dataset_remote["masks"], masks_local_dir)

    def run(self):
        for dataset_name in ("WFM", "SIM", "SXT", "Cryo-ET"):
            print(f"Fetching {dataset_name} dataset...")
            self.fetch_dataset(dataset_name)


def main(config_path: str | None = None):
    DataFetcher(config_path).run()
