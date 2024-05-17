from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

import numpy as np

from musc.commons import explain_object


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


class BaseModel(Generic[XR, X, YR, Y, YP], ABC):

    @abstractmethod
    def predict(self, x: X) -> YP:
        pass

    def inner(self) -> Any:
        return self

    def clone(self) -> BaseModel[XR, X, YR, Y, YP]:
        return deepcopy(self)

    def clone_with_resource(self, resource: Any) -> BaseModel[XR, X, YR, Y, YP]:
        del resource
        return self.clone()

    def resource_current(self) -> Any:
        return None

    def resource_cpu(self) -> Any:
        return None

    def preprocess_x(self, x: XR) -> X:
        # Implementors should ensure that the behavior of
        # this method will not be changed by model updator.
        return x  # type: ignore

    def preprocess_y(self, y: YR) -> Y:
        # Implementors should ensure that the behavior of
        # this method will not be changed by model updator.
        return y  # type: ignore

    def y_pred_to_json_like(self, y_pred: YP) -> Any:
        del y_pred
        raise NotImplementedError


@dataclass
class UpdateStrategy(Generic[X, Y, YP]):

    drift_detector: BaseDriftDetector[X, Y, YP]
    updator: Updator[X, Y]
    data_amount_required: int
    use_ncl: bool

    def __post_init__(self) -> None:
        if self.data_amount_required < 0:
            raise TypeError
        if self.use_ncl and not isinstance(self.drift_detector, BaseDriftDetectorWithNcl):
            raise TypeError

    def clone_without_state(self) -> UpdateStrategy[X, Y, YP]:
        return UpdateStrategy(
            self.drift_detector.clone_without_state(),
            self.updator,
            self.data_amount_required,
            self.use_ncl,
        )


class BaseDriftDetector(Generic[X, Y, YP], ABC):

    @abstractmethod
    def step(self, x: X, y: Y, y_pred: YP) -> DriftDetected | None:
        pass

    def reset(self) -> None:
        raise NotImplementedError

    def clone_without_state(self) -> BaseDriftDetector[X, Y, YP]:
        cloned = deepcopy(self)
        cloned.reset()
        return cloned


class BaseDriftDetectorWithNcl(Generic[X, Y, YP], BaseDriftDetector[X, Y, YP]):

    def clone_without_state(self) -> BaseDriftDetectorWithNcl[X, Y, YP]:
        cloned = deepcopy(self)
        cloned.reset()
        return cloned


class Metric(Generic[Y, YP]):

    @overload
    def __init__(self, callable: Callable[[Y, YP], Any], pred_first: Literal[False]) -> None:
        pass

    @overload
    def __init__(self, callable: Callable[[YP, Y], Any], pred_first: Literal[True]) -> None:
        pass

    def __init__(self, callable: Callable[[Any, Any], Any], pred_first: bool) -> None:
        self._callable = callable
        self._pred_first = pred_first
        try:
            from torch import Tensor
            self._torch_tensor_class = Tensor
        except ImportError:
            self._torch_tensor_class = None

    def __call__(self, *, y: Y, y_pred: YP) -> float:
        if self._pred_first:
            metric_value = self._callable(y_pred, y)
        else:
            metric_value = self._callable(y, y_pred)
        if (
            self._torch_tensor_class is not None
            and isinstance(metric_value, self._torch_tensor_class)
        ):
            return metric_value.tolist()  # type: ignore
        if isinstance(metric_value, np.ndarray):
            return metric_value.tolist()
        return metric_value

    def __explain_self__(self) -> str:
        self_class_explained = explain_object(__class__)
        callable_explained = explain_object(self._callable)
        pred_first_explained = explain_object(self._pred_first)
        return f'{self_class_explained}(callable={callable_explained}, pred_first={pred_first_explained})'


@dataclass
class DriftDetected:
    ncl: int | None


# TODO: hide `BaseModel` here
Updator = Callable[[BaseModel, list[X], list[Y]], None]


__all__ = [
    'BaseModel',
    'UpdateStrategy',
    'BaseDriftDetector',
    'BaseDriftDetectorWithNcl',
    'Metric',
    'DriftDetected',
    'Updator',
]
