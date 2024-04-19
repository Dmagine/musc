from collections.abc import Iterable, Sequence
from typing import Any, Literal, overload

from musc.evaluator.a import Evaluator as EvaluatorLowLevel
from musc.evaluator.a2 import Evaluator2 as EvaluatorLowLevel2
from musc.service.concepts.a import Metric, UpdateStrategy
from musc.service.concepts.high_level.a import apply_model_adaptors
from musc.service.impl.dd_process.stats import Stats


class Evaluator:

    def __init__(
        self,
        model: Any,
        update_strategy: UpdateStrategy,
        mode: Literal['simulation'] | Literal['idle_detect'] = 'simulation',
    ) -> None:
        model_ = apply_model_adaptors(model)
        update_strategy_ = update_strategy.clone_without_state()
        match mode:
            case 'simulation':
                self._inner = EvaluatorLowLevel2(model_, update_strategy_)
            case 'idle_detect':
                self._inner = EvaluatorLowLevel(model_, update_strategy_)
            case _:
                raise ValueError

    @overload
    def evaluate(
        self,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
    ) -> tuple[list[Any], Stats]:
        pass

    @overload
    def evaluate(
        self,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
        metric: Metric,
    ) -> tuple[float, Stats]:
        pass

    @overload
    def evaluate(
        self,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
        metric: Sequence[Metric],
    ) -> tuple[list[float], Stats]:
        pass

    def evaluate(
        self,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
        metric: Metric | Sequence[Metric] | None = None,
    ) -> tuple[float | list[float] | list[Any], Stats]:

        x_arr_, y_arr_, t_x_arr_, t_y_arr_ = list(x_arr), list(y_arr), list(t_x_arr), list(t_y_arr)
        del x_arr, y_arr, t_x_arr, t_y_arr
        if not len(x_arr_) == len(y_arr_) == len(t_x_arr_) == len(t_y_arr_) != 0:
            raise ValueError

        if metric is None:
            metric_ = None
        elif isinstance(metric, Metric):
            metric_ = metric
        else:
            metric_ = list(metric)
        del metric

        y_pred_arr, stats = self._inner.evaluate(x_arr_, y_arr_, t_x_arr_, t_y_arr_)
        if metric_ is None:
            return y_pred_arr, stats
        elif isinstance(metric_, Metric):
            return self._score(y_arr_, y_pred_arr, metric_), stats
        else:
            return [self._score(y_arr_, y_pred_arr, m) for m in metric_], stats

    def _score(self, y_arr: list[Any], y_pred_arr: list[Any], metric: Metric) -> float:
        model = self._inner.get_model()
        return sum(
            metric(y=model.preprocess_y(y), y_pred=y_pred)
            for y, y_pred in zip(y_arr, y_pred_arr, strict=True)
        ) / len(y_arr)


__all__ = [
    'Evaluator',
]
