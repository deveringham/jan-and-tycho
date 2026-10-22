"""Animation output: MP4 via ffmpeg when available, GIF via Pillow otherwise.

Never fail a run because ffmpeg is missing (SPEC §2).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation


@lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and mpl_animation.writers.is_available("ffmpeg")


@dataclass(frozen=True)
class AnimationSettings:
    """Frame budget: default ≤ 20 s at 30 fps; `quick` renders short, low-res versions."""

    fps: int = 30
    max_seconds: float = 20.0
    dpi: int = 100
    quick: bool = False
    force_gif: bool = False

    @property
    def n_frames(self) -> int:
        if self.quick:
            return 12
        return int(self.fps * self.max_seconds)

    @property
    def effective_fps(self) -> int:
        return 6 if self.quick else self.fps

    @property
    def effective_dpi(self) -> int:
        return 50 if self.quick else self.dpi

    def frames(self, requested: int | None = None) -> int:
        n = self.n_frames if requested is None else min(requested, self.n_frames)
        return max(n, 2)


def save_animation(anim: FuncAnimation, out_stem: str | Path, settings: AnimationSettings) -> Path:
    """Write `anim` to `<out_stem>.mp4` (ffmpeg) or `<out_stem>.gif` (Pillow fallback)."""
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fps = settings.effective_fps
    if ffmpeg_available() and not settings.force_gif:
        path = stem.with_suffix(".mp4")
        writer = mpl_animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            # yuv420p (needed by most players) requires even frame dimensions.
            extra_args=["-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
        )
    else:
        path = stem.with_suffix(".gif")
        writer = mpl_animation.PillowWriter(fps=fps)
    # Drop a stale render in the other format so outputs never mix MP4 and GIF.
    other = stem.with_suffix(".gif" if path.suffix == ".mp4" else ".mp4")
    other.unlink(missing_ok=True)
    anim.save(str(path), writer=writer, dpi=settings.effective_dpi)
    plt.close(anim._fig)  # pyright: ignore[reportAttributeAccessIssue]
    return path
