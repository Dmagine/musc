from __future__ import annotations

from typing import Any

from river.base import BinaryDriftDetector, DriftDetector

from musc.service.concepts.a import BaseDriftDetector, BaseDriftDetectorWithNcl, DriftDetected, Metric


class DriftDetectorByRiver(BaseDriftDetector):

    def __init__(self, drift_detector: DriftDetector, metric: Metric) -> None:
        self._drift_detector = drift_detector
        self._metric = metric

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(self._metric(y=y, y_pred=y_pred))
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


class DriftDetectorByRiverWithManualNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: DriftDetector, metric: Metric, ncl: int) -> None:
        self._drift_detector = drift_detector
        self._metric = metric
        self._ncl = ncl
        self._cnt = 0

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._cnt += 1
        self._drift_detector.update(self._metric(y=y, y_pred=y_pred))
        if self._drift_detector.drift_detected:
            if self._ncl <= -1:
                return DriftDetected(self._cnt)
            else:
                return DriftDetected(min(self._ncl, self._cnt))
        else:
            return None

    def clone_without_state(self) -> DriftDetectorByRiverWithManualNcl:
        return DriftDetectorByRiverWithManualNcl(
            self._drift_detector.clone(),
            self._metric,
            self._ncl,
        )


class DriftDetectorByRiverBinaryWithManualNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: BinaryDriftDetector, ncl: int) -> None:
        self._drift_detector = drift_detector
        self._ncl = ncl
        self._cnt = 0

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._cnt += 1
        self._drift_detector.update(y != y_pred)
        if self._drift_detector.drift_detected:
            if self._ncl <= -1:
                return DriftDetected(self._cnt)
            else:
                return DriftDetected(min(self._ncl, self._cnt))
        else:
            return None

    def clone_without_state(self) -> DriftDetectorByRiverBinaryWithManualNcl:
        return DriftDetectorByRiverBinaryWithManualNcl(self._drift_detector.clone(), self._ncl)


class DriftDetectorByRiverWithAutoNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: DriftDetector, metric: Metric) -> None:
        assert hasattr(drift_detector, 'new_concept_length')
        self._drift_detector = drift_detector
        self._metric = metric

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(self._metric(y=y, y_pred=y_pred))
        ncl = self._drift_detector.new_concept_length  # type: ignore
        return DriftDetected(ncl) if ncl is not None else None

    def clone_without_state(self) -> DriftDetectorByRiverWithAutoNcl:
        return DriftDetectorByRiverWithAutoNcl(self._drift_detector.clone(), self._metric)


class DriftDetectorByRiverBinaryWithAutoNcl(BaseDriftDetectorWithNcl):

    def __init__(self, drift_detector: BinaryDriftDetector) -> None:
        assert hasattr(drift_detector, 'new_concept_length')
        self._drift_detector = drift_detector

    def step(self, x: Any, y: Any, y_pred: Any) -> DriftDetected | None:
        del x
        self._drift_detector.update(y != y_pred)
        ncl = self._drift_detector.new_concept_length  # type: ignore
        return DriftDetected(ncl) if ncl is not None else None

    def clone_without_state(self) -> DriftDetectorByRiverBinaryWithAutoNcl:
        return DriftDetectorByRiverBinaryWithAutoNcl(self._drift_detector.clone())


__all__ = [
    'DriftDetectorByRiver',
    'DriftDetectorByRiverBinary',
    'DriftDetectorByRiverWithAutoNcl',
    'DriftDetectorByRiverBinaryWithAutoNcl',
]
