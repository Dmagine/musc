from __future__ import annotations

import logging
import typing
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Generic, TypeVar

from readerwriterlock.rwlock import RWLockFair

from musc.service.concepts.a import BaseModel, Updator


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


class ModelWrapper(Generic[XR, X, YR, Y, YP]):

    def __init__(
        self,
        model: BaseModel[XR, X, YR, Y, YP],
        updator: Updator[X, Y],
        daemon: bool,
        debug: bool,
    ) -> None:
        self._model = model
        self._gen = 0
        self._updator = updator
        self._daemon = daemon
        self._debug = debug
        self._model_lock = RWLockFair()
        self._update_lock = Lock()

    def gen(self) -> int:
        with self._model_lock.gen_rlock():
            gen = self._gen
        return gen

    def predict(self, x: X) -> ModelPredictionProcess[X, YP]:
        with self._model_lock.gen_rlock():
            model = self._model
            gen = self._gen
        return ModelPredictionProcess(model, gen, x, self._daemon, self._debug)

    def update_by(self, x_arr: list[X], y_arr: list[Y]) -> None:
        with self._update_lock:
            model_new = self._model.clone()
            try:
                self._updator(model_new, x_arr, y_arr)
            except Exception as e:
                if self._debug:
                    raise
                logging.error(f'The model updator raised an exception: {repr(e)}')
                with self._model_lock.gen_wlock():
                    self._gen += 1
            else:
                with self._model_lock.gen_wlock():
                    self._model = model_new
                    self._gen += 1

    def preprocess_x(self, x: XR) -> X:
        with self._model_lock.gen_rlock():
            model = self._model
        return model.preprocess_x(x)

    def preprocess_y(self, y: YR) -> Y:
        with self._model_lock.gen_rlock():
            model = self._model
        return model.preprocess_y(y)


class ModelPredictionProcess(Generic[X, YP]):

    def __init__(
        self,
        model: BaseModel[XR, X, YR, Y, YP],
        gen: int,
        x: X,
        daemon: bool,
        debug: bool,
    ) -> None:

        self._model = model
        self._gen = gen
        self._x = x
        self._debug = debug

        self._success = None
        self._y_pred = None

        self._thread = Thread(target=self._thread_target, daemon=daemon)
        self._thread.start()

    def _thread_target(self) -> None:
        try:
            y_pred = self._model.predict(self._x)
        except Exception as e:
            if self._debug:
                raise
            self._success = False
            self._y_pred = e
        else:
            self._success = True
            self._y_pred = y_pred

    def gen(self) -> int:
        return self._gen

    def join(self) -> Prediction[YP]:
        self._thread.join()
        assert isinstance(self._success, bool)
        return Prediction(self._success, typing.cast(YP | Exception, self._y_pred), self._gen)


@dataclass
class Prediction(Generic[YP]):
    success: bool
    y_pred: YP | Exception
    gen: int


__all__ = [
    'ModelWrapper',
    'ModelPredictionProcess',
    'Prediction',
]
