from .dataset import SIM, SXT, WFM, CryoET, LoadingMode

for cls in (SIM, SXT, WFM, CryoET):
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
        dataset = cls(mode)
        memory = dataset.get_memory_usage()
        print(f"  {desc}: {memory['total_mb']:.2f} MB")
    print()
