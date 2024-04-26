import copy
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Generic, TypeVar

from musc.eval.active_count import BaseActiveCount
from musc.service.concepts.a import UpdateStrategy
from musc.service.impl.model_wrapper import ModelWrapper
from musc.service.impl.data_history.base import BaseDataHistory
from musc.service.impl.dd_process.dh_walker import DataHistoryWalker, Never
from musc.service.impl.dd_process.stats import ModelUpdateRecord, Stats, StatsWithLock


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')
X_, Y_ = TypeVar('X_'), TypeVar('Y_')
T = TypeVar('T')


class DriftDetectorProcess(Generic[X, Y, YP]):

    @dataclass
    class ModeDetecting:
        pass

    @dataclass
    class ModeCollecting(Generic[X_, Y_]):
        x_arr: list[X_]
        y_arr: list[Y_]

    def __init__(
        self,
        data_history: BaseDataHistory[X, Y, YP],
        update_strategy: UpdateStrategy[X, Y, YP],
        model: ModelWrapper[XR, X, YR, Y, YP],
        wait_y_timeout: float | None,
        daemon: bool,
        active_count: BaseActiveCount,
    ) -> None:

        self._data_history = data_history
        self._update_strategy = update_strategy
        self._model = model

        self._data_history_walker = DataHistoryWalker(
            self._data_history,
            self._model,
            wait_y_timeout,
            active_count,
        )
        self._mode = DriftDetectorProcess.ModeDetecting()
        self._last_drift_point = None
        self._last_new_concept_range_start = None
        self._stats = StatsWithLock()

        self._active_count = active_count
        self._thread = Thread(target=self._thread_target, daemon=daemon)
        self._active_count.inc()
        self._thread.start()

    def _thread_target(self) -> None:

        while True:

            if isinstance(self._mode, DriftDetectorProcess.ModeDetecting):

                step_result = self._data_history_walker.step()
                if isinstance(step_result, Never):
                    break
                x, y, y_pred = step_result

                if (ncl := self._update_strategy.drift_detector.step(x, y, y_pred)) is None:
                    continue
                if self._update_strategy.use_ncl:
                    ncl = ncl.ncl
                    if ncl is None or ncl < 0:
                        raise RuntimeError
                else:
                    ncl = 0

                x_arr, y_arr = list[X](), list[Y]()
                self._last_new_concept_range_start = self._data_history_walker.cnt()
                self._last_drift_point = self._data_history_walker.cnt()
                with self._data_history.gen_rlock():
                    while True:
                        if len(x_arr) == ncl:
                            break
                        if self._last_new_concept_range_start == 0:
                            raise RuntimeError
                        if self._last_new_concept_range_start - 1 \
                                not in self._data_history_walker.err_indices():
                            cur_item = self._data_history[self._last_new_concept_range_start - 1]
                            x_arr.append(cur_item.x)
                            y_arr.append(typing.cast(Y, cur_item.y))
                        self._last_new_concept_range_start -= 1
                x_arr.reverse()
                y_arr.reverse()

                if len(x_arr) < self._update_strategy.data_amount_required:
                    self._mode = DriftDetectorProcess.ModeCollecting(x_arr, y_arr)
                else:
                    self._update(x_arr, y_arr)

            elif isinstance(self._mode, DriftDetectorProcess.ModeCollecting):

                step_result = self._data_history_walker.step(require_y_pred=False)
                if isinstance(step_result, Never):
                    break
                x, y, _ = step_result

                self._mode.x_arr.append(x)
                self._mode.y_arr.append(y)

                if len(self._mode.x_arr) == self._update_strategy.data_amount_required:
                    self._update(self._mode.x_arr, self._mode.y_arr)
                    self._mode = DriftDetectorProcess.ModeDetecting()

            else:

                assert False

        self._active_count.dec()

    def _update(self, x_arr: list[X], y_arr: list[Y]) -> None:
        assert self._last_drift_point is not None
        assert self._last_new_concept_range_start is not None
        start_time = time.time()
        self._model.update_by(x_arr, y_arr)
        time_spent = time.time() - start_time
        with self._stats.gen_wlock():
            self._stats.inner().model_update_records.append(
                ModelUpdateRecord(
                    self._last_drift_point,
                    (self._last_new_concept_range_start, self._data_history_walker.cnt()),
                    time_spent,
                )
            )

    def stats(self, fn: Callable[[Stats], T] = lambda s: s) -> T:
        with self._stats.gen_rlock():
            return copy.deepcopy(fn(self._stats.inner()))

    def stop_listening(self) -> None:
        self._data_history_walker.stop_listening()

    def join(self) -> None:
        return self._thread.join()


__all__ = [
    'DriftDetectorProcess',
]
