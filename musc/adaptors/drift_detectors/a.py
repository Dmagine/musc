from __future__ import annotations

import random
from typing import Any

from musc.service.concepts.a import BaseDriftDetectorWithNcl, DriftDetected


class DriftDetectorNone(BaseDriftDetectorWithNcl):

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x, y, y_pred
        return None

    def clone_without_state(self) -> DriftDetectorNone:
        return DriftDetectorNone()


class DriftDetectorPeriodicallyUpdate(BaseDriftDetectorWithNcl):

    def __init__(
        self,
        period: int,
        ncl: int | None = None,
        probability: float | None = None,
    ) -> None:
        assert period >= 1
        assert probability is None or 0.0 <= probability <= 1.0
        self._period = period
        self._ncl = ncl if ncl is not None else period
        self._cnt = 0
        self._probability = probability

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x, y, y_pred
        self._cnt += 1
        if self._cnt % self._period == 0:
            if self._probability is not None and random.random() >= self._probability:
                return None
            if self._ncl <= -1:
                return DriftDetected(self._cnt)
            else:
                return DriftDetected(min(self._ncl, self._cnt))
        else:
            return None

    def clone_without_state(self) -> DriftDetectorPeriodicallyUpdate:
        return DriftDetectorPeriodicallyUpdate(self._period, self._ncl, self._probability)


__all__ = [
    'DriftDetectorNone',
    'DriftDetectorPeriodicallyUpdate',
]
