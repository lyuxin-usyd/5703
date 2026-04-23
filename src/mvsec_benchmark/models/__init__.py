"""Model utilities for MVSEC benchmark."""

from .dummy_flow import DummyFlowHead
from .linear_flow import LinearFlowRegressor

try:
    from .evflownet_like import EVFlowNetLike
except Exception:  # pragma: no cover - optional torch dependency
    EVFlowNetLike = None

__all__ = ["DummyFlowHead", "LinearFlowRegressor", "EVFlowNetLike"]
