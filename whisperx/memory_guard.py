"""Unified memory management for Apple Silicon MLX + PyTorch.

Clearing caches is advisory; peak RAM must be measured and checked separately.
"""
from __future__ import annotations
import gc
import logging
import psutil

log = logging.getLogger("whisperx.memory")


def memory_snapshot() -> dict:
    vm = psutil.virtual_memory()
    values = {
        "available_gb": round(vm.available / 1024**3, 3),
        "used_percent": vm.percent,
        "process_rss_gb": round(psutil.Process().memory_info().rss / 1024**3, 3),
    }
    try:
        import mlx.core as mx
        values["mlx_active_gb"] = round(mx.metal.get_active_memory() / 1024**3, 3)
        values["mlx_peak_gb"] = round(mx.metal.get_peak_memory() / 1024**3, 3)
    except (ImportError, AttributeError, RuntimeError):
        pass
    return values


def get_free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1024**3


def release_mlx_model() -> None:
    """Remove mlx-whisper ModelHolder's persistent reference, when present."""
    try:
        from importlib import import_module
        module = import_module("mlx_whisper.transcribe")
        holder = getattr(module, "ModelHolder", None)
        if holder is not None:
            holder.model = None
            holder.model_path = None
    except (ImportError, AttributeError):
        pass


def cleanup_unified_memory(force: bool = True) -> None:
    if force:
        gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    try:
        import mlx.core as mx
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass
    if force:
        gc.collect()


def check_memory_safety(threshold_gb: float = 1.2) -> bool:
    available = get_free_ram_gb()
    if available < threshold_gb:
        cleanup_unified_memory()
        available = get_free_ram_gb()
    if available < threshold_gb:
        raise MemoryError(
            f"Insufficient macOS unified memory: {available:.2f} GB free, "
            f"minimum {threshold_gb:.2f} GB"
        )
    return True
