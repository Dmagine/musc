from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from river.base import BinaryDriftDetector, DriftDetector

from musc.adaptors.drift_detectors.a import DriftDetectorNone, DriftDetectorPeriodicallyUpdate
from musc.adaptors.drift_detectors.river.a import (
    DriftDetectorByRiver,
    DriftDetectorByRiverBinary,
    DriftDetectorByRiverBinaryWithNcl,
    DriftDetectorByRiverWithNcl,
)
from musc.adaptors.drift_detectors.river.with_kdp import ADWIN, PageHinkley, kswin_distr_prelude
from musc.hpo.distr.a import BaseDistribution
from musc.hpo.distr.high_level import into_distr
from musc.service.concepts.a import BaseModel, Metric, UpdateStrategy


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


class ModelGuard(Generic[XR, X, YR, Y, YP], BaseModel[XR, X, YR, Y, YP]):

    def predict(self, x: X) -> YP:
        del x
        raise RuntimeError

    def inner(self) -> Any:
        raise RuntimeError

    def clone(self) -> BaseModel[XR, X, YR, Y, YP]:
        raise RuntimeError

    def clone_with_resource(self, resource: Any) -> BaseModel[XR, X, YR, Y, YP]:
        del resource
        raise RuntimeError

    def resource_current(self) -> Any:
        raise RuntimeError

    def resource_cpu(self) -> Any:
        raise RuntimeError

    def preprocess_x(self, x: XR) -> X:
        del x
        raise RuntimeError

    def preprocess_y(self, y: YR) -> Y:
        del y
        raise RuntimeError

    def y_pred_to_json_like(self, y_pred: YP) -> Any:
        del y_pred
        raise RuntimeError


class UpdateStrategySubclass(UpdateStrategy):

    def __repr__(self) -> str:
        return UpdateStrategy(
            self.drift_detector,
            self.updator,
            self.data_amount_required,
            self.use_ncl,
        ).__repr__()


class UpdateStrategyNone(UpdateStrategySubclass):

    def __init__(self) -> None:
        super().__init__(DriftDetectorNone(), updator_no_op, 0, False)

    def clone_without_state(self) -> UpdateStrategyNone:
        return UpdateStrategyNone()


class UpdateStrategyByPeriodicallyUpdate(UpdateStrategySubclass):

    def __init__(
        self,
        period: int,
        updator: Callable[[Any, list[Any], list[Any]], None],
        *,
        ncl: int | None = None,
        probability: float | None = None,
    ) -> None:
        super().__init__(
            DriftDetectorPeriodicallyUpdate(period, ncl, probability),
            UpdatorWrapper(updator),
            period,
            True,
        )


class UpdateStrategyByDriftDetection(UpdateStrategySubclass):

    def __init__(
        self,
        drift_detector: BinaryDriftDetector | DriftDetector,
        updator: Callable[[Any, list[Any], list[Any]], None],
        data_amount_required: int,
        metric: Metric | None = None,
        use_ncl: bool | None = None,
    ) -> None:
        if isinstance(drift_detector, BinaryDriftDetector):
            if not hasattr(drift_detector, 'new_concept_length'):
                drift_detector_ = DriftDetectorByRiverBinary(drift_detector)
            else:
                drift_detector_ = DriftDetectorByRiverBinaryWithNcl(drift_detector)
        elif isinstance(drift_detector, DriftDetector):
            if metric is None:
                raise TypeError
            if not hasattr(drift_detector, 'new_concept_length'):
                drift_detector_ = DriftDetectorByRiver(drift_detector, metric)
            else:
                drift_detector_ = DriftDetectorByRiverWithNcl(drift_detector, metric)
        else:
            raise TypeError
        drift_detector_ = drift_detector_.clone_without_state()
        updator_ = UpdatorWrapper(updator)
        use_ncl = use_ncl if use_ncl is not None else hasattr(drift_detector, 'new_concept_length')
        super().__init__(drift_detector_, updator_, data_amount_required, use_ncl)

    @staticmethod
    def kwargs_distr_prelude() -> dict[str, BaseDistribution[Any]]:
        return {
            'drift_detector': into_distr([
                {'type': ADWIN},
                kswin_distr_prelude(),
                {'type': PageHinkley},
            ]),
        }


class UpdatorWrapper:

    def __init__(self, inner: Callable[[Any, list[Any], list[Any]], None]) -> None:
        self._inner = inner

    def __call__(self, model: BaseModel, x_arr: list[Any], y_arr: list[Any]) -> None:
        self._inner(model.inner(), x_arr, y_arr)


def apply_model_adaptors(model: Any) -> BaseModel:

    if isinstance(model, BaseModel):
        return model

    try:
        import torch
    except ImportError:
        pass
    else:
        from musc.adaptors.models import ModelPyTorch
        if isinstance(model, torch.nn.Module):
            return ModelPyTorch(model)

    try:
        import sklearn
    except ImportError:
        pass
    else:
        from musc.adaptors.models import ModelScikitLearn
        if isinstance(model, sklearn.base.BaseEstimator):
            return ModelScikitLearn(model)

    raise TypeError('The model is not recognized by musc')


def updator_no_op(model: BaseModel, x_arr: list[Any], y_arr: list[Any]) -> None:
    del model, x_arr, y_arr


__all__ = [
    'ModelGuard',
    'UpdateStrategyNone',
    'UpdateStrategyByPeriodicallyUpdate',
    'UpdateStrategyByDriftDetection',
    'apply_model_adaptors',
]
