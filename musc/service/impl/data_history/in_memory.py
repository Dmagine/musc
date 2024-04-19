from __future__ import annotations

import typing
from typing import Generic, TypeVar

from readerwriterlock.rwlock import RWLockFair

from musc.evaluator.active_count import BaseActiveCount
from musc.service.errors import ServiceErrorByDuplicatedId, ServiceErrorFromModelPrediction
from musc.service.impl.data_history.base import (
    BaseDataHistory,
    BaseDataHistoryListener,
    DataHistoryItem,
    DataHistoryItemYNone,
)
from musc.service.impl.model_wrapper import ModelPredictionProcess, ModelWrapper, Prediction


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


class DataHistoryInMemory(Generic[X, Y, YP], BaseDataHistory[X, Y, YP]):

    def __init__(
        self,
        model: ModelWrapper[XR, X, YR, Y, YP],
        active_count: BaseActiveCount,
    ) -> None:

        self._model = model

        self._core = DataHistoryInMemoryCore[X, Y, YP]()
        self._id2idx = dict[str, int]()
        self._y_waiting_room = dict[str, Y]()

        self._listener: BaseDataHistoryListener[X, Y, YP] | None = None
        self._active_count = active_count

        self._lock = RWLockFair()

    def __len__(self) -> int:
        return len(self._core)

    def __getitem__(self, idx: int) -> DataHistoryItem[X, Y, YP]:
        return self._core[idx]

    def gen_rlock(self) -> RWLockFair._aReader:
        return self._lock.gen_rlock()

    def _gen_wlock(self) -> RWLockFair._aWriter:
        return self._lock.gen_wlock()

    def recv_x(self, x: X, id_: str) -> YP:
        with self._gen_wlock():
            if id_ in self._id2idx:
                raise ServiceErrorByDuplicatedId
            idx = len(self._core)
            y = self._y_waiting_room.pop(id_, DataHistoryItemYNone())
            y_pred = self._model.predict(x)
            self._id2idx[id_] = idx
            self._core.append(x, y, y_pred)
            listener = self._listener
        if listener is not None:
            listener.notify_x(idx, x, y_pred)
            if not isinstance(y, DataHistoryItemYNone):
                listener.notify_y(idx, y)
        y_pred = y_pred.join()
        with self._gen_wlock():
            self._core.recv_y_pred(idx, y_pred)
        if not y_pred.success:
            raise ServiceErrorFromModelPrediction(typing.cast(Exception, y_pred.y_pred))
        return typing.cast(YP, y_pred.y_pred)

    def recv_y(self, y: Y, id_: str) -> None:
        with self._gen_wlock():
            idx = self._id2idx.get(id_)
            if idx is None:
                if id_ in self._y_waiting_room:
                    raise ServiceErrorByDuplicatedId
                self._y_waiting_room[id_] = y
                listener_and_idx = None
            else:
                if not self._core.recv_y(idx, y):
                    raise ServiceErrorByDuplicatedId
                listener_and_idx = self._listener, idx
        if listener_and_idx is not None:
            listener, idx = listener_and_idx
            if listener is not None:
                listener.notify_y(idx, y)

    def register_listener(self, listener: BaseDataHistoryListener[X, Y, YP] | None) -> None:
        with self._gen_wlock():
            self._listener = listener


class DataHistoryInMemoryCore(Generic[X, Y, YP]):

    def __init__(self) -> None:
        self._items = list[DataHistoryItem[X, Y, YP]]()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DataHistoryItem[X, Y, YP]:
        return self._items[idx]

    def append(self, x: X, y: Y | DataHistoryItemYNone, y_pred: ModelPredictionProcess[X, YP]) -> None:
        self._items.append(DataHistoryItem(x, y, y_pred))

    def recv_y(self, idx: int, y: Y) -> bool:
        if not 0 <= idx < len(self._items):
            raise RuntimeError
        if not isinstance(self._items[idx].y, DataHistoryItemYNone):
            return False
        self._items[idx].y = y
        return True

    def recv_y_pred(self, idx: int, y_pred: Prediction[YP]) -> bool:
        if not 0 <= idx < len(self._items):
            raise RuntimeError
        if not isinstance(self._items[idx].y_pred, ModelPredictionProcess):
            return False
        self._items[idx].y_pred = y_pred
        return True


__all__ = [
    'DataHistoryInMemory',
]
