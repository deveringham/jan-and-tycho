"""The observer's position as Jan knows it from `PublicMeta`.

Under T1 the observer's orbit is published; Jan propagates it with generic conic code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from jt_core.conics import ConicElements, elements_to_state
from jt_core.schema import PublicMeta

FloatArray = NDArray[np.float64]


def observer_function(meta: PublicMeta) -> Callable[[FloatArray], FloatArray]:
    """t → observer heliocentric positions (N, 3). Raises if the tier hides the observer."""
    if meta.observer_elements is not None:
        el = ConicElements(*meta.observer_elements.as_tuple())
        gm = meta.gm_sun
        return lambda t: elements_to_state(el, np.atleast_1d(np.asarray(t, dtype=np.float64)), gm)[
            0
        ]
    if meta.observer_position is not None:
        pos = np.asarray(meta.observer_position, dtype=np.float64)
        return lambda t: np.broadcast_to(pos, (np.size(t), 3)).copy()
    raise ValueError(f"tier {meta.tier}: observer is not known to Jan")
