"""Polygon recognition for kinematic paths: straight edges intersected into vertices.

Triangulated points cannot exist where the body was never observable (e.g. a corner that
always falls near solar conjunction); joining neighbouring points then cuts the corner.
If the points lie on a few straight edges, fitting a line per edge and intersecting
neighbouring lines recovers the corners anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _rdp(points: FloatArray, eps: float) -> list[int]:
    """Ramer–Douglas–Peucker on an open polyline: indices of the kept points."""
    a, b = points[0], points[-1]
    ab = b - a
    n = float(np.linalg.norm(ab))
    if len(points) < 3:
        return [0, len(points) - 1]
    if n == 0:
        d = np.linalg.norm(points - a, axis=1)
    else:
        d = np.linalg.norm(np.cross(points - a, ab / n), axis=1)
    k = int(np.argmax(d))
    if d[k] <= eps:
        return [0, len(points) - 1]
    left = _rdp(points[: k + 1], eps)
    right = _rdp(points[k:], eps)
    return left[:-1] + [i + k for i in right]


def _line(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Robust 3D line: PCA, then refit without points > 3 × the median distance."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    d = vt[0]
    for _ in range(2):
        resid = np.linalg.norm(np.cross(points - c, d), axis=1)
        keep = resid <= 3 * max(float(np.median(resid)), 1e-12)
        if keep.sum() < 2:
            break
        c = points[keep].mean(axis=0)
        _, _, vt = np.linalg.svd(points[keep] - c, full_matrices=False)
        d = vt[0]
    return c, d


def _closest(c1: FloatArray, d1: FloatArray, c2: FloatArray, d2: FloatArray) -> FloatArray:
    """Midpoint of the shortest segment between two 3D lines."""
    w = c1 - c2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-12:
        return 0.5 * (c1 + c2)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((c1 + s * d1) + (c2 + t * d2))


@dataclass
class Polygon:
    vertices: FloatArray
    """(V, 3), in path order."""
    after_index: list[int]
    """For each vertex, the index of the last ordered point before it."""
    rms: float
    """RMS distance of the points to the polygon (AU)."""


def _dist_to_polygon(points: FloatArray, verts: FloatArray) -> FloatArray:
    best = np.full(len(points), np.inf)
    for a, b in zip(verts, np.roll(verts, -1, axis=0), strict=True):
        ab = b - a
        t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-30), 0, 1)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * ab), axis=1))
    return best


def _segments(n: int, keys: list[int]) -> list[list[int]]:
    """Point indices of each edge between consecutive corner keys (closed loop)."""
    out = []
    for a, b in zip(keys, [*keys[1:], keys[0] + n], strict=True):
        out.append([j % n for j in range(a, b + 1)])
    return out


def _merged_residual(points: FloatArray, keys: list[int], drop: int) -> float:
    """RMS line residual of the edge formed by removing corner `drop`."""
    m = len(keys)
    a, b = keys[drop - 1], keys[(drop + 1) % m]
    n = len(points)
    idx = [j % n for j in range(a, b + (n if b <= a else 0) + 1)]
    c, d = _line(points[idx])
    return float(np.sqrt(np.mean(np.linalg.norm(np.cross(points[idx] - c, d), axis=1) ** 2)))


def fit_polygon(
    points: FloatArray,
    eps: float,
    max_vertices: int = 12,
    merge_angle_deg: float = 15.0,
    min_edge_points: int = 4,
) -> Polygon | None:
    """Fit a closed polygon to phase-ordered points; None if no small polygon fits.

    RDP proposes corners; edges that are nearly collinear with a neighbour, or carry too
    few points to define a line, are merged before the final vertices are intersected.
    """
    n = len(points)
    if n < 8:
        return None
    i1 = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    loop = np.concatenate([points, points[:1]])
    keys = sorted(set(_rdp(loop[: i1 + 1], eps) + [i + i1 for i in _rdp(loop[i1:], eps)]) - {n})
    cos_merge = np.cos(np.radians(merge_angle_deg))
    while len(keys) > 3:
        segs = _segments(n, keys)
        lines = [_line(points[sg]) for sg in segs]
        short = [k for k in range(len(keys)) if len(segs[k]) - 2 < min_edge_points]
        if short:
            # Too few points to define edge k: drop whichever of its two corners leaves
            # the straighter merged edge.
            k = min(short, key=lambda j: len(segs[j]))
            cands = [k, (k + 1) % len(keys)]
            keys.pop(min(cands, key=lambda c: _merged_residual(points, keys, c)))
            continue
        worst, worst_par = None, cos_merge
        for k in range(len(keys)):
            par = abs(float(lines[k][1] @ lines[k - 1][1]))
            if par > worst_par:
                worst, worst_par = k, par
        if worst is None:
            break
        keys.pop(worst)  # drop the corner between edge worst-1 and edge worst
    if not 3 <= len(keys) <= max_vertices:
        return None
    segs = _segments(n, keys)
    edges = [_line(points[sg[1:-1]] if len(sg) > 3 else points[sg]) for sg in segs]
    verts = []
    after = []
    for k, key in enumerate(keys):
        prev, cur = edges[k - 1], edges[k]
        verts.append(_closest(prev[0], prev[1], cur[0], cur[1]))
        after.append((key - 1) % n)
    v = np.array(verts)
    rms = float(np.sqrt(np.mean(_dist_to_polygon(points, v) ** 2)))
    return Polygon(v, after, rms)
