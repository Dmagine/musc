from __future__ import annotations

import typing
from numbers import Number
from typing import Any

from river.base import BinaryDriftDetector, DriftDetector

from musc.service.concepts.a import BaseDriftDetector, BaseDriftDetectorWithNcl, DriftDetected, Metric


class DriftDetectorByRiver(BaseDriftDetector):

    def __init__(self, drift_detector: DriftDetector, metric: Metric) -> None:
        self._drift_detector = drift_detector
        self._metric = metric

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        # https://github.com/python/mypy/issues/3186
        self._drift_detector.update(typing.cast(Number, self._metric(y=y, y_pred=y_pred)))
        return DriftDetected(None) if self._drift_detector.drift_detected else None

    def clone_without_state(self) -> DriftDetectorByRiver:
        return DriftDetectorByRiver(self._drift_detector.clone(), self._metric)


class DriftDetectorByRiverBinary(BaseDriftDetector):

    def __init__(self, drift_detector: BinaryDriftDetector) -> None:
        self._drift_detector = drift_detector

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(y != y_pred)
        return DriftDetected(None) if self._drift_detector.drift_detected else None

    def clone_without_state(self) -> DriftDetectorByRiverBinary:
        return DriftDetectorByRiverBinary(self._drift_detector.clone())


class DriftDetectorByRiverWithNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: DriftDetector, metric: Metric) -> None:
        assert hasattr(drift_detector, 'new_concept_length')
        self._drift_detector = drift_detector
        self._metric = metric

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(typing.cast(Number, self._metric(y=y, y_pred=y_pred)))
        ncl = self._drift_detector.new_concept_length  # type: ignore
        return DriftDetected(ncl) if ncl is not None else None

    def clone_without_state(self) -> DriftDetectorByRiverWithNcl:
        return DriftDetectorByRiverWithNcl(self._drift_detector.clone(), self._metric)


class DriftDetectorByRiverBinaryWithNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: BinaryDriftDetector) -> None:
        assert hasattr(drift_detector, 'new_concept_length')
        self._drift_detector = drift_detector

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(y != y_pred)
        ncl = self._drift_detector.new_concept_length  # type: ignore
        return DriftDetected(ncl) if ncl is not None else None

    def clone_without_state(self) -> DriftDetectorByRiverBinaryWithNcl:
        return DriftDetectorByRiverBinaryWithNcl(self._drift_detector.clone())


__all__ = [
    'DriftDetectorByRiver',
    'DriftDetectorByRiverBinary',
    'DriftDetectorByRiverWithNcl',
    'DriftDetectorByRiverBinaryWithNcl',
]
