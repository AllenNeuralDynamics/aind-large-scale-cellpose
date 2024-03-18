"""
Defines all the types used in the module
"""

from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

PathLike = Union[Path, str]
ArrayLike = Union[da.Array, np.ndarray]
