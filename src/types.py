from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

    type Array = np.ndarray | cp.ndarray
else:
    type Array = np.ndarray
