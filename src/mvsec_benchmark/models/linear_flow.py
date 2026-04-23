from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearFlowRegressor:
    ridge: float = 1e-3
    add_bias: bool = True
    weights: np.ndarray | None = None

    def _design_matrix(self, representation: np.ndarray) -> np.ndarray:
        rep = np.asarray(representation, dtype=np.float32)
        if rep.ndim != 3:
            raise ValueError(f"Expected representation shape (C,H,W), got {rep.shape}")
        feat = np.moveaxis(rep, 0, -1).reshape(-1, rep.shape[0])
        if self.add_bias:
            feat = np.concatenate([feat, np.ones((feat.shape[0], 1), dtype=np.float32)], axis=1)
        return feat

    def fit(self, representations: list[np.ndarray], gt_flows: list[np.ndarray]) -> "LinearFlowRegressor":
        if len(representations) != len(gt_flows):
            raise ValueError("representations and gt_flows must have the same length")
        if not representations:
            raise ValueError("At least one training sample is required")

        x_all = []
        y_all = []
        for rep, flow in zip(representations, gt_flows):
            x = self._design_matrix(rep)
            y = np.asarray(flow, dtype=np.float32).reshape(-1, 2)
            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Pixel count mismatch: {x.shape} vs {flow.shape}")
            x_all.append(x)
            y_all.append(y)

        x_mat = np.concatenate(x_all, axis=0)
        y_mat = np.concatenate(y_all, axis=0)
        xtx = x_mat.T @ x_mat
        xty = x_mat.T @ y_mat
        reg = self.ridge * np.eye(xtx.shape[0], dtype=np.float32)
        self.weights = np.linalg.solve(xtx + reg, xty).astype(np.float32, copy=False)
        return self

    def predict(self, representation: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet.")
        rep = np.asarray(representation, dtype=np.float32)
        h, w = rep.shape[1:]
        x = self._design_matrix(rep)
        pred = x @ self.weights
        return pred.reshape(h, w, 2).astype(np.float32, copy=False)
