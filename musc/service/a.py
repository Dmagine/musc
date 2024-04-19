from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from musc.evaluator.active_count import BaseActiveCount, NullActiveCount
from musc.service.concepts.a import BaseModel, UpdateStrategy
from musc.service.impl.model_wrapper import ModelWrapper
from musc.service.impl.data_history.base import BaseDataHistory
from musc.service.impl.data_history.in_memory import DataHistoryInMemory
from musc.service.impl.dd_process.a import DriftDetectorProcess
from musc.service.impl.dd_process.stats import Stats


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')
T = TypeVar('T')


class Service(Generic[XR, X, YR, Y, YP]):

    def __init__(
        self,
        model: BaseModel[XR, X, YR, Y, YP],
        update_strategy: UpdateStrategy[X, Y, YP],
        *,
        wait_y_timeout: float | None = None,
        daemon: bool = True,
        debug: bool = False,
        active_count: BaseActiveCount = NullActiveCount(),
    ) -> None:
        if not isinstance(active_count, NullActiveCount) and wait_y_timeout is not None:
            raise NotImplementedError(
                'Evaluating service with `wait_y_timeout` is currently not supported.'
            )
        self._model = ModelWrapper(model, update_strategy.updator, daemon, debug)
        self._data_history: BaseDataHistory[X, Y, YP] \
            = DataHistoryInMemory(self._model, active_count)
        self._drift_detector_process = DriftDetectorProcess(
            self._data_history,
            update_strategy,
            self._model,
            wait_y_timeout,
            daemon,
            active_count,
        )

    def recv_x(self, x: XR, id_: str) -> YP:
        return self._data_history.recv_x(self._model.preprocess_x(x), id_)

    def recv_y(self, y: YR, id_: str) -> None:
        return self._data_history.recv_y(self._model.preprocess_y(y), id_)

    def stats(self, fn: Callable[[Stats], T] = lambda s: s) -> T:
        return self._drift_detector_process.stats(fn)

    def stop_listening(self) -> None:
        self._drift_detector_process.stop_listening()

    def join(self) -> None:
        self._drift_detector_process.join()


__all__ = [
    'Service',
]
