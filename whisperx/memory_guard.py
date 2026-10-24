"""
Memory guard for Apple Silicon Unified Memory.
"""

import gc
import psutil
import platform


def get_free_ram_gb() -> float:
    """Get available RAM in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)


def cleanup_unified_memory(force: bool = True):
    """Aggressive Unified Memory buffer cleanup."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    try:
        import mlx.core as mx
        if hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    except:
        pass
    if force:
        gc.collect()


def check_memory_safety(threshold_gb: float = 2.0) -> bool:
    """Check if system has enough free RAM."""
    free_gb = psutil.virtual_memory().available / (1024 ** 3)
    if free_gb < threshold_gb:
        print(f"Warning: Low memory ({free_gb:.2f}GB), forcing cleanup...")
        cleanup_unified_memory(force=True)
    return free_gb >= threshold_gb
