"""Device and precision selection.

device    auto -> CUDA > MPS > CPU  (env ZHJUDGE_DEVICE=cpu forces CPU even when a GPU/MPS exists)
precision auto -> CUDA compute capability >= 8 (Ampere+: A100/A10/L4/3090/4090/H100) : bf16 autocast
                  CUDA compute capability < 8  (T4 = 7.5, V100 = 7.0)                   : fp16 autocast + GradScaler
                  MPS                                                                  : bf16 autocast (as the spike)
                  CPU                                                                  : fp32
Master weights and optimizer state are always fp32.
"""
from __future__ import annotations

import os

import torch


def select_device(pref: str | None = "auto") -> torch.device:
    pref = (os.environ.get("ZHJUDGE_DEVICE") or pref or "auto").lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("device cuda requested but torch.cuda.is_available() is False "
                         "(no GPU runtime, or a torch wheel without CUDA / for another CUDA version)")
    if pref == "mps" and not mps_ok:
        raise SystemExit("device mps requested but MPS is not available")
    return torch.device(pref)


def select_precision(device: torch.device, pref: str | None = "auto") -> str:
    pref = (os.environ.get("ZHJUDGE_PRECISION") or pref or "auto").lower()
    if device.type == "cpu":
        if pref not in ("auto", "fp32"):
            print(f"[precision] {pref} requested on CPU; using fp32", flush=True)
        return "fp32"
    if pref != "auto":
        if pref == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
            print("[precision] bf16 not supported natively on this GPU; using fp16 + GradScaler", flush=True)
            return "fp16"
        return pref
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return "bf16" if major >= 8 else "fp16"
    return "bf16"  # mps


def amp_dtype(precision: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def want_grad_checkpointing(setting, device: torch.device) -> bool:
    """auto: on for CUDA GPUs with < 20 GiB (T4 16 GB, V100 16 GB), off otherwise."""
    if isinstance(setting, bool):
        return setting
    if device.type != "cuda":
        return False
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    return total < 20


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.0f} GiB, sm_{p.major}{p.minor}), torch {torch.__version__}"
    return f"{device.type}, torch {torch.__version__}"


def set_threads(n: int):
    n = int(os.environ.get("ZHJUDGE_THREADS") or n or 0)
    if n > 0:
        torch.set_num_threads(n)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_gib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    try:
        import resource
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (2**30 if sys.platform == "darwin" else 2**20)
    except Exception:  # noqa: BLE001
        return 0.0
