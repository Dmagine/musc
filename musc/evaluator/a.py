import operator
from collections.abc import Iterable
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Any

from musc.evaluator.active_count import ActiveCount
from musc.service.a import Service
from musc.service.concepts.a import BaseModel, UpdateStrategy
from musc.service.impl.dd_process.stats import Stats


class Evaluator:

    def __init__(
        self,
        model: BaseModel,
        update_strategy: UpdateStrategy,
        debug: bool = False,
    ) -> None:
        self._active_count, self._idle_event = ActiveCount.create()
        self._service = Service(
            model,
            update_strategy,
            daemon=False,
            debug=debug,
            active_count=self._active_count,
        )
        self._thread_pool = ThreadPool()
        self._evaluated = [False, Lock()]

    def get_model(self) -> BaseModel:
        with self._service._model._model_lock.gen_rlock():
            return self._service._model._model

    def evaluate(
        self,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
    ) -> tuple[list[Any], Stats]:
        with self._evaluated[1]:
            if self._evaluated[0]:
                raise RuntimeError
            self._evaluated[0] = True
        events = []
        for i, (x, t_x) in enumerate(zip(x_arr, t_x_arr)):
            events.append((t_x, 'x', x, i))
        for i, (y, t_y) in enumerate(zip(y_arr, t_y_arr)):
            events.append((t_y, 'y', y, i))
        events.sort(key=operator.itemgetter(0))
        return self._evaluate(events)

    def _evaluate(
        self,
        events: list[tuple[float, str, Any, int]],
    ) -> tuple[list[Any], Stats]:
        last_t = None
        results = [None] * (len(events) // 2)
        results_lock = Lock()
        for t, ty, v, i in events:
            if last_t is None:
                self._idle_event.wait()
            else:
                self._idle_event.wait(t - last_t)
            last_t = t
            if ty == 'x':
                self._active_count.inc()
                self._thread_pool.apply_async(self._recv_x, (v, i, results, results_lock))
            elif ty == 'y':
                self._active_count.inc()
                self._thread_pool.apply_async(self._recv_y, (v, i))
            else:
                assert False
        self._service.stop_listening()
        self._service.join()
        self._thread_pool.close()
        self._thread_pool.join()
        return results, self._service.stats()

    def _recv_x(
        self,
        x: Any,
        id_: int,
        results: list[Any | None],
        result_lock: Lock,
    ) -> None:
        result = self._service.recv_x(x, str(id_))
        self._active_count.dec()
        with result_lock:
            results[id_] = result

    def _recv_y(self, y: Any, id_: int) -> None:
        self._service.recv_y(y, str(id_))
        self._active_count.dec()


__all__ = [
    'Evaluator',
]
