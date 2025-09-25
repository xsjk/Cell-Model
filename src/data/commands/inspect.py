from ..dataset import SIM, SXT, WFM, CryoET, LoadingMode


def format_bytes(bytes_value) -> str:
    # Format bytes to human readable format
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def main(config_path: str | None = None):
    for cls in (WFM, SIM, SXT, CryoET):
        name = cls.dataset_name
        dataset = cls()
        print(f"=== {name} Dataset ===")
        shape_info = dataset.shape_info
        print(f"Size: {shape_info['num_samples']} samples")
        print(f"Image shape: {shape_info['image_shape']}")
        print(f"Mask shape: {shape_info['mask_shape']}")

        print("Memory Usage:")
        for mode, desc in {
            LoadingMode.ON_DEMAND: "On Demand",
            LoadingMode.COMPRESSED_CACHE: "Compressed Cache",
            LoadingMode.FULL_MEMORY: "Full Memory",
        }.items():
            dataset = cls(loading_mode=mode, config_path=config_path)
            memory = dataset.get_memory_usage()
            print(f"  {desc}: {format_bytes(memory['total'])}")
        print()
