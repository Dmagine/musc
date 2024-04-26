from __future__ import annotations

import queue
import typing
from queue import Queue
from threading import Lock
from typing import Generic, Literal, TypeVar, overload

from musc.eval.active_count import BaseActiveCount, NullActiveCount
from musc.service.impl.model_wrapper import ModelPredictionProcess, ModelWrapper, Prediction
from musc.service.impl.data_history.base import (
    BaseDataHistory,
    BaseDataHistoryListener,
    DataHistoryItemYNone,
)


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')
X_, Y_, YP_ = TypeVar('X_'), TypeVar('Y_'), TypeVar('YP_')


class DataHistoryWalker(Generic[X, Y, YP]):

    class Listener(Generic[X_, Y_, YP_], BaseDataHistoryListener[X_, Y_, YP_]):

        def __init__(self, active_count: BaseActiveCount) -> None:
            self._x_queues = dict[int, Queue[tuple[X_, ModelPredictionProcess[X_, YP_]] | Never]]()
            self._y_queues = dict[int, Queue[Y_ | Never]]()
            self._stopped = False
            self._active_count = active_count
            self._lock = Lock()

        def request_x(self, idx: int) -> Queue[tuple[X_, ModelPredictionProcess[X_, YP_]] | Never]:
            with self._lock:
                if idx in self._x_queues:
                    raise RuntimeError
                queue = Queue[tuple[X_, ModelPredictionProcess[X_, YP_]] | Never]()
                if self._stopped:
                    self._active_count.inc()
                    queue.put(Never())
                else:
                    self._x_queues[idx] = queue
                return queue

        def request_y(self, idx: int) -> Queue[Y_ | Never]:
            with self._lock:
                if idx in self._y_queues:
                    raise RuntimeError
                queue = Queue[Y_ | Never]()
                if self._stopped:
                    self._active_count.inc()
                    queue.put(Never())
                else:
                    self._y_queues[idx] = queue
                return queue

        def notify_x(self, idx: int, x: X_, y_pred: ModelPredictionProcess[X_, YP_]) -> None:
            with self._lock:
                if idx in self._x_queues:
                    self._active_count.inc()
                    self._x_queues.pop(idx).put((x, y_pred))

        def notify_y(self, idx: int, y: Y_) -> None:
            with self._lock:
                if idx in self._y_queues:
                    self._active_count.inc()
                    self._y_queues.pop(idx).put(y)

        def stop(self) -> None:
            with self._lock:
                self._stopped = True
                for idx in list(self._x_queues.keys()):
                    self._active_count.inc()
                    self._x_queues.pop(idx).put(Never())
                for idx in list(self._y_queues.keys()):
                    self._active_count.inc()
                    self._y_queues.pop(idx).put(Never())

    def __init__(
        self,
        data_history: BaseDataHistory[X, Y, YP],
        model: ModelWrapper[XR, X, YR, Y, YP],
        wait_y_timeout: float | None,
        active_count: BaseActiveCount,
    ) -> None:
        assert isinstance(active_count, NullActiveCount) or wait_y_timeout is None
        self._data_history = data_history
        self._model = model
        self._wait_y_timeout = wait_y_timeout
        self._active_count = active_count
        self._listener = DataHistoryWalker.Listener[X, Y, YP](active_count)
        self._data_history.register_listener(self._listener)
        self._cnt = 0
        self._err_indices = set[int]()

    def _wait_x(self) -> tuple[X, ModelPredictionProcess[X, YP] | Prediction[YP]] | Never:
        with self._data_history.gen_rlock():
            if self._cnt < len(self._data_history):
                cur_item = self._data_history[self._cnt]
                return cur_item.x, cur_item.y_pred
            queue = self._listener.request_x(self._cnt)
        self._active_count.dec()
        return queue.get()

    def _wait_y_after_x(self) -> Y | DataHistoryItemYNone | Never:
        with self._data_history.gen_rlock():
            if not isinstance(y := self._data_history[self._cnt].y, DataHistoryItemYNone):
                return y
            queue_ = self._listener.request_y(self._cnt)
        self._active_count.dec()
        try:
            return queue_.get(timeout=self._wait_y_timeout)
        except queue.Empty:
            return DataHistoryItemYNone()

    @overload
    def step(self) -> tuple[X, Y, YP] | Never:
        pass

    @overload
    def step(self, require_y_pred: Literal[True]) -> tuple[X, Y, YP] | Never:
        pass

    @overload
    def step(self, require_y_pred: Literal[False]) -> tuple[X, Y, None] | Never:
        pass

    def step(self, require_y_pred: bool = True) -> tuple[X, Y, YP | None] | Never:
        while True:
            wait_result = self._wait_x()
            if isinstance(wait_result, Never):
                return wait_result
            x, y_pred = wait_result
            if require_y_pred:
                if isinstance(y_pred, ModelPredictionProcess):
                    if y_pred.gen() != self._model.gen():
                        y_pred = self._model.predict(x).join()
                    else:
                        y_pred = y_pred.join()
                elif isinstance(y_pred, Prediction):
                    if y_pred.gen != self._model.gen():
                        y_pred = self._model.predict(x).join()
                else:
                    assert False
                if not y_pred.success:
                    self._err_indices.add(self._cnt)
                    self._cnt += 1
                    continue
                y_pred = typing.cast(YP, y_pred.y_pred)
            else:
                y_pred = None
            wait_result = self._wait_y_after_x()
            if isinstance(wait_result, Never):
                return wait_result
            if isinstance(wait_result, DataHistoryItemYNone):
                self._err_indices.add(self._cnt)
                self._cnt += 1
                continue
            y = typing.cast(Y, wait_result)
            self._cnt += 1
            return x, y, y_pred

    def cnt(self) -> int:
        return self._cnt

    def err_indices(self) -> set[int]:
        return self._err_indices

    def stop_listening(self) -> None:
        self._listener.stop()


class Never:
    pass


__all__ = [
    'DataHistoryWalker',
    'Never',
]
