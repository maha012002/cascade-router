"""Keep the laptop responsive during embedding / training.

Caps BLAS/OpenMP thread storms (which freeze UI when RAM is tight) and refuses to
start a heavy job if free RAM is below a threshold.
"""

from __future__ import annotations

import os
import sys

# Set before numpy/sklearn/lightgbm import wherever possible. Call early in CLIs.
DEFAULT_THREADS = 2


def limit_threads(n: int = DEFAULT_THREADS) -> None:
    """Cap math-library thread pools so we don't pin every core."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(n))
    # LightGBM also reads this.
    os.environ.setdefault("LIGHTGBM_NUM_THREADS", str(n))


def free_ram_gb() -> float | None:
    """Available physical RAM in GiB, or None if the OS won't say."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
        except Exception:
            return None
    return None


def require_free_ram(min_gb: float, what: str) -> None:
    """Abort with a clear message if free RAM is too low for ``what``."""
    free = free_ram_gb()
    if free is None:
        print(f"warning: could not measure free RAM before {what}; proceeding carefully")
        return
    print(f"free RAM before {what}: {free:.1f} GiB (need >= {min_gb:.1f})")
    if free < min_gb:
        raise SystemExit(
            f"Refusing to start {what}: only {free:.1f} GiB free RAM "
            f"(need at least {min_gb:.1f}).\n"
            "Close Chrome/Edge tabs, other IDEs, Docker, games, etc., then retry.\n"
            "Using a GPU would not fix this — the process needs free system RAM."
        )
