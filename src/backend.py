from dataclasses import dataclass
from types import ModuleType

import numpy as np

from src.logger import get_logger
from src.types import Array

logger = get_logger(__name__)


@dataclass(frozen=True)
class BackendPolicy:
    use_gpu: bool = False
    min_gpu_bytes: int = 8_000_000


class BackendManager:
    xp: ModuleType

    def __init__(self):
        self.xp = np
        self.name = "numpy"
        self._cp = None

    def _load_cupy(self):
        if self._cp is None:
            import cupy as cp  # noqa: PLC0415

            self._cp = cp
        return self._cp

    def _gpu_available(self, cp_mod) -> bool:
        try:
            return cp_mod.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def _cupy_runtime_ok(self, cp_mod) -> bool:
        try:
            if cp_mod.cuda.runtime.getDeviceCount() <= 0:
                return False
            from cupy.cuda import nvrtc  # noqa: PLC0415

            _ = nvrtc.getVersion()  # type: ignore
            _ = cp_mod.exp(cp_mod.asarray([0.0], dtype=cp_mod.float32))
            return True
        except Exception:
            return False

    def configure(self, policy: BackendPolicy, dataset_bytes: int):
        use_cupy = False

        if policy.use_gpu and dataset_bytes >= policy.min_gpu_bytes:
            try:
                cp_mod = self._load_cupy()
                use_cupy = self._cupy_runtime_ok(cp_mod)
                logger.info(f"GPU requested. CuPy available: {use_cupy}.")
            except Exception:
                use_cupy = False
                logger.info(f"GPU requested. CuPy available: {use_cupy}.")

        if use_cupy:
            self.xp = self._cp
            self.name = "cupy"
        else:
            self.xp = np
            self.name = "numpy"
        logger.info(f"Using backend: {self.name}")
        return self

    def asarray(self, x: Array, dtype=None) -> Array:
        return self.xp.asarray(x, dtype=dtype)

    def to_numpy(self, x: Array) -> np.ndarray:
        if self.name == "cupy" and hasattr(self._cp, "asnumpy"):
            return self._cp.asnumpy(x)
        return x


backend = BackendManager()
