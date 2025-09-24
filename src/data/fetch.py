import os
import os.path
import re
import socket

import socks
from tqdm import tqdm

from .config import load_config
from .filesystem import FTPFS

config = load_config()
remote_config = config["Remote"]
paths_config = config["Paths"]

# Set up proxy if specified
if "proxy" in remote_config:
    proxy_url = remote_config["proxy"]
    if match := re.match(r"^(socks5|http)://([^:/]+):(\d+)$", proxy_url):
        scheme, host, port = match.groups()
        proxy_type = socks.SOCKS5 if scheme == "socks5" else socks.HTTP
        socks.set_default_proxy(proxy_type, host, int(port))
        socket.socket = socks.socksocket
    else:
        raise ValueError(f"Invalid proxy URL format: {proxy_url}")

# Extract FTP connection config from Remote section
ftp_config = {
    "host": remote_config["host"],
    "port": remote_config["port"],
    "user": remote_config["user"],
    "passwd": remote_config["passwd"],
}
ftp = FTPFS(**ftp_config)


def fetch_all(remote_pattern: str, local_dir: str):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    for path in tqdm(ftp.ls(remote_pattern), desc=f"Downloading {remote_pattern} to {local_dir}"):
        remote_dir, filename = os.path.split(path)
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            continue
        ftp.read_numpy(path, save=local_path)


def fetch_dataset(dataset_name: str):
    if dataset_name not in remote_config["datasets"]:
        raise ValueError(f"Dataset {dataset_name} not found in configuration")

    # Load configurations
    dataset_remote = remote_config["datasets"][dataset_name]
    dataset_paths = paths_config["datasets"][dataset_name]

    # Build local paths
    images_local_dir = os.path.join(paths_config["original_dir"], dataset_paths["images"])
    masks_local_dir = os.path.join(paths_config["original_dir"], dataset_paths["masks"])

    # Fetch images
    images_patterns = dataset_remote["images"]
    if isinstance(images_patterns, list):
        for pattern in images_patterns:
            fetch_all(pattern, images_local_dir)
    else:
        assert isinstance(images_patterns, str)
        fetch_all(images_patterns, images_local_dir)

    # Fetch masks
    fetch_all(dataset_remote["masks"], masks_local_dir)


# Fetch all datasets
for dataset_name in ["WFM", "SIM", "SXT", "Cryo-ET"]:
    print(f"Fetching {dataset_name} dataset...")
    fetch_dataset(dataset_name)
