from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Generic, TypeVar

from musc.service.impl.model_wrapper import ModelPredictionProcess, Prediction


X, Y, YP = TypeVar('X'), TypeVar('Y'), TypeVar('YP')


class BaseDataHistory(Generic[X, Y, YP], ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DataHistoryItem[X, Y, YP]:
        pass

    @abstractmethod
    def gen_rlock(self) -> AbstractContextManager:
        pass

    @abstractmethod
    def recv_x(self, x: X, id_: str) -> YP:
        pass

    @abstractmethod
    def recv_y(self, y: Y, id_: str) -> None:
        pass

    @abstractmethod
    def register_listener(self, listener: BaseDataHistoryListener[X, Y, YP] | None) -> None:
        pass


@dataclass
class DataHistoryItem(Generic[X, Y, YP]):
    x: X
    y: Y | DataHistoryItemYNone
    y_pred: ModelPredictionProcess[X, YP] | Prediction[YP]


class DataHistoryItemYNone:
    pass


class BaseDataHistoryListener(Generic[X, Y, YP], ABC):

    @abstractmethod
    def notify_x(self, idx: int, x: X, y_pred: ModelPredictionProcess[X, YP]) -> None:
        pass

    @abstractmethod
    def notify_y(self, idx: int, y: Y) -> None:
        pass


__all__ = [
    'BaseDataHistory',
    'DataHistoryItem',
    'DataHistoryItemYNone',
    'BaseDataHistoryListener',
]
