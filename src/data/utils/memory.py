import numpy as np


def get_memory_size(obj) -> int:
    if isinstance(obj, (str, bytes)):
        return len(obj)
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_memory_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_memory_size(k) + get_memory_size(v) for k, v in obj.items())
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
