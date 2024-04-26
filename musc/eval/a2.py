from __future__ import annotations

import operator
import time
from collections.abc import Callable
from typing import Generic, TypeVar

from musc.service.concepts.a import BaseModel, UpdateStrategy
from musc.service.impl.dd_process.a import DriftDetectorProcess
from musc.service.impl.dd_process.stats import ModelUpdateRecord, Stats


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')
XR_, X_, YR_, Y_, YP_ = TypeVar('XR_'), TypeVar('X_'), TypeVar('YR_'), TypeVar('Y_'), TypeVar('YP_')
V = TypeVar('V')


class Evaluator2(Generic[XR, X, YR, Y, YP]):

    def __init__(
        self,
        model: BaseModel[XR, X, YR, Y, YP],
        update_strategy: UpdateStrategy[X, Y, YP],
    ) -> None:

        self.__model_history = [(model, float('-inf'), int(0))]
        self.__update_strategy = update_strategy

        self.__evaluated = False

        self.__y_pred_arr = list[tuple[YP, float, int] | None]()

        self._current_time = float('-inf')
        self.__mode = DriftDetectorProcess.ModeDetecting()
        self.__last_drift_point = None
        self.__last_new_concept_range_start = None
        self.__stats = Stats([])

    def get_model(self) -> BaseModel[XR, X, YR, Y, YP]:
        return self.__model_history[-1][0]

    def evaluate(
        self,
        x_arr: list[XR],
        y_arr: list[YR],
        t_x_arr: list[float],
        t_y_arr: list[float],
    ) -> tuple[list[YP], Stats]:

        if not len(x_arr) == len(y_arr) == len(t_x_arr) == len(t_y_arr):
            raise ValueError

        if self.__evaluated:
            raise RuntimeError
        self.__evaluated = True

        x_arr_preprocessed = list[X]()
        t_x_arr_preprocessed = list[float]()
        for x, t_x in zip(x_arr, t_x_arr, strict=True):
            x_p, t_p = self.timeit(lambda: self.__model_history[0][0].preprocess_x(x))
            x_arr_preprocessed.append(x_p)
            t_x_arr_preprocessed.append(t_x + t_p)

        y_arr_preprocessed = list[Y]()
        t_y_arr_preprocessed = list[float]()
        for y, t_y in zip(y_arr, t_y_arr, strict=True):
            y_p, t_p = self.timeit(lambda: self.__model_history[0][0].preprocess_y(y))
            y_arr_preprocessed.append(y_p)
            t_y_arr_preprocessed.append(t_y + t_p)

        return self.__evaluate(
            x_arr_preprocessed,
            y_arr_preprocessed,
            t_x_arr_preprocessed,
            t_y_arr_preprocessed,
        )

    def __evaluate(
        self,
        x_arr: list[X],
        y_arr: list[Y],
        t_x_arr: list[float],
        t_y_arr: list[float],
    ) -> tuple[list[YP], Stats]:

        data = zip(x_arr, y_arr, t_x_arr, t_y_arr, range(len(x_arr)), strict=True)
        data = sorted(data, key=operator.itemgetter(2))
        del x_arr, y_arr, t_x_arr, t_y_arr

        self.__y_pred_arr.extend([None] * len(data))
        data_cursor = 0

        for cnt, (x, y, t_x, t_y, id_) in enumerate(data, start=1):
            while data_cursor < len(data):
                x_, _, t_x_, _, id__ = data[data_cursor]
                if t_x_ > max(t_x, t_y):
                    break
                self.__prune_model_history_by_t_x(t_x_)
                y_pred_, t_p_ = self.timeit(lambda: self.__model_history[0][0].predict(x_))
                self.__y_pred_arr[id__] = (y_pred_, t_p_, self.__model_history[0][2])
                data_cursor += 1
            if isinstance(self.__mode, DriftDetectorProcess.ModeDetecting):
                y_pred = self.__determine_t_dhw_step(x, t_x, t_y, id_)
                with self.Timer(self) as timer:
                    if (ncl := self.__update_strategy.drift_detector.step(x, y, y_pred)) is None:
                        continue
                    if self.__update_strategy.use_ncl:
                        ncl = ncl.ncl
                        if ncl is None or ncl < 0:
                            raise RuntimeError
                    else:
                        ncl = 0
                    data_ = data[cnt-ncl:cnt]
                    x_arr_, y_arr_ = [d[0] for d in data_], [d[1] for d in data_]
                    self.__last_new_concept_range_start, self.__last_drift_point = cnt - ncl, cnt
                    if len(x_arr_) < self.__update_strategy.data_amount_required:
                        self.__mode = DriftDetectorProcess.ModeCollecting(x_arr_, y_arr_)
                    else:
                        self.__update(x_arr_, y_arr_, cnt, timer)
            elif isinstance(self.__mode, DriftDetectorProcess.ModeCollecting):
                self.__determine_t_dhw_step_woyp(t_x, t_y)
                with self.Timer(self) as timer:
                    self.__mode.x_arr.append(x)
                    self.__mode.y_arr.append(y)
                    if len(self.__mode.x_arr) == self.__update_strategy.data_amount_required:
                        self.__update(self.__mode.x_arr, self.__mode.y_arr, cnt, timer)
                        self.__mode = DriftDetectorProcess.ModeDetecting()
            else:
                assert False

        y_pred_arr = list[YP]()
        for item in self.__y_pred_arr:
            assert item is not None
            y_pred_arr.append(item[0])
        return y_pred_arr, self.__stats

    def __determine_t_dhw_step(self, x: X, t_x: float, t_y: float, id_: int) -> YP:
        assert (y_pred_item := self.__y_pred_arr[id_]) is not None
        y_pred, t_p, gen = y_pred_item
        if gen == self.__model_history[-1][2]:
            self._current_time = max(self._current_time, t_x + t_p, t_y)
        else:
            y_pred, t_p = self.timeit(lambda: self.__model_history[-1][0].predict(x))
            self._current_time = max(max(self._current_time, t_x) + t_p, t_y)
        return y_pred

    def __determine_t_dhw_step_woyp(self, t_x: float, t_y: float) -> None:
        self._current_time = max(self._current_time, t_x, t_y)

    def __prune_model_history_by_t_x(self, t_x: float) -> None:
        for i in reversed(range(len(self.__model_history))):
            if self.__model_history[i][1] <= t_x:
                del self.__model_history[:i]
                return
        assert False

    def __update(
        self,
        x_arr: list[X],
        y_arr: list[Y],
        cnt: int,
        timer: Timer[XR, X, YR, Y, YP],
    ) -> None:
        assert self.__last_drift_point is not None
        assert self.__last_new_concept_range_start is not None
        _, time_spent = self.timeit(lambda: self.__update_by(x_arr, y_arr, timer))
        self.__stats.model_update_records.append(ModelUpdateRecord(
            self.__last_drift_point,
            (self.__last_new_concept_range_start, cnt),
            time_spent,
        ))

    def __update_by(self, x_arr: list[X], y_arr: list[Y], timer: Timer[XR, X, YR, Y, YP]) -> None:
        model_new = self.__model_history[-1][0].clone()
        self.__update_strategy.updator(model_new, x_arr, y_arr)
        self.__model_history.append((model_new, timer.get(), self.__model_history[-1][2] + 1))

    @staticmethod
    def timeit(callable_: Callable[[], V]) -> tuple[V, float]:
        start_time = time.time()
        return callable_(), time.time() - start_time

    class Timer(Generic[XR_, X_, YR_, Y_, YP_]):

        def __init__(self, evaluator: Evaluator2[XR_, X_, YR_, Y_, YP_]) -> None:
            self.__evaluator = evaluator
            self.__start_time = None

        def get(self) -> float:
            if self.__start_time is None:
                raise RuntimeError
            return self.__evaluator._current_time + (time.time() - self.__start_time)

        def __enter__(self) -> Evaluator2.Timer[XR_, X_, YR_, Y_, YP_]:
            self.__start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            self.__evaluator._current_time = self.get()
            self.__start_time = None


__all__ = [
    'Evaluator2',
]
