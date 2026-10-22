"""3D trajectory errors on the dense grid (SPEC §7, level "Trajectory")."""

from __future__ import annotations

from typing import Any

import numpy as np

from rudolf.data import RunView


def trajectory_metrics(view: RunView) -> dict[str, Any]:
    keys = ("traj_rms_train_au", "traj_max_train_au", "traj_rms_holdout_au", "traj_max_holdout_au")
    pred = view.dense_pred_r
    if pred is None:
        return dict.fromkeys(keys)
    err = np.linalg.norm(pred - view.dense_true_r, axis=1)
    train = view.dense_t < view.sealed.t_split
    return {
        "traj_rms_train_au": float(np.sqrt(np.mean(err[train] ** 2))),
        "traj_max_train_au": float(np.max(err[train])),
        "traj_rms_holdout_au": float(np.sqrt(np.mean(err[~train] ** 2))),
        "traj_max_holdout_au": float(np.max(err[~train])),
    }
