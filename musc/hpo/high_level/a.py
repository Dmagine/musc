from __future__ import annotations

import itertools
import logging
import re
import time
import typing
from collections.abc import Callable, Hashable, Iterable, Sequence
from multiprocessing import Manager
from os import PathLike
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Generic, Literal, TypeVar

import pandas as pd

from musc.commons import safe_save
from musc.evaluator.high_level import Evaluator
from musc.hpo.a import ExhaustiveSearch, RandomizedSearch, Sample
from musc.hpo.distr.a import BaseDistribution
from musc.hpo.distr.high_level import into_distr
from musc.hpo.high_level.helper import ModelCloneHelper
from musc.hpo.res_pool.base import BaseResourcePool
from musc.hpo.res_pool.global_ import GlobalResourcePool
from musc.service.concepts.a import BaseModel, Metric, UpdateStrategy
from musc.service.concepts.high_level.a import apply_model_adaptors
from musc.service.impl.dd_process.stats import Stats


S = TypeVar('S')


class BaseUpdateStrategySearch(Generic[S]):

    def __init__(
        self,
        distr: BaseDistribution[UpdateStrategy],
        model: BaseModel,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
        metric: list[Metric],
        score_fn: Callable[[list[float], Stats], S],
        greater_score_fn: Callable[[S, S], bool] | None,
        weak_greater_score_fn: Callable[[S, S], bool] | None,
        eval_retry: int,
        top_k_kept: int,
        exhaustive: bool,
        n_jobs: int | None,
        resource_pool: BaseResourcePool[Any] | None,
        sample_callback: Callable[[Sample[UpdateStrategy, None]], None] | None,
        optimal_candidates_callback: Callable[
            [
                Sample[UpdateStrategy, S],
                list[Sample[UpdateStrategy, S]],
                set[Hashable] | None,
                int | None,
            ],
            bool,
        ] | None,
        suboptimal_candidates_callback: Callable[
            [Sample[UpdateStrategy, S], bool, set[Hashable] | None, int | None],
            bool,
        ] | None,
        top_k_callback: \
            Callable[[Sample[UpdateStrategy, S], list[Sample[UpdateStrategy, S]]], None] | None,
        debug: bool,
    ) -> None:

        eval_max_n_tries = eval_retry + 1 if eval_retry >= 0 else None

        # Sending CUDA model to other processes is problematic.
        # See https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
        # for details.
        resource_initial = model.resource_current()
        self._model_clone_helper = ModelCloneHelper(model.clone_with_resource(model.resource_cpu()))
        model = self._model_clone_helper.portable()

        if resource_pool is not None:
            self._pool = GlobalResourcePool(resource_pool)
            global_resource_pool = self._pool.portable()
        else:
            self._pool = global_resource_pool = None

        def score_fn_(us: UpdateStrategy) -> S:
            if not isinstance(us, UpdateStrategy):
                raise TypeError
            score, stats = None, None
            for i in range(eval_max_n_tries) if eval_max_n_tries is not None else itertools.count():
                try:
                    if global_resource_pool is None:
                        score, stats = Evaluator(model.clone_with_resource(resource_initial), us) \
                            .evaluate(x_arr, y_arr, t_x_arr, t_y_arr, metric)
                    else:
                        with global_resource_pool.alloc() as resource:
                            score, stats = Evaluator(model.clone_with_resource(resource), us) \
                                .evaluate(x_arr, y_arr, t_x_arr, t_y_arr, metric)
                except Exception as e:
                    if debug:
                        raise
                    eval_max_n_tries_ = eval_max_n_tries if eval_max_n_tries is not None else 'inf'
                    logging.warn(
                        f'The evaluation process raised an exception: {repr(e)}'
                        f' (number of tries: {i+1}/{eval_max_n_tries_})'
                    )
                    time.sleep(60.0)
                else:
                    break
            if score is None or stats is None:
                raise RuntimeError('Failed to execute evaluation')
            return score_fn(score, stats)

        if not exhaustive:
            if optimal_candidates_callback is not None:
                optimal_candidates_callback_ \
                    = lambda s, o, t: optimal_candidates_callback(s, o, t, None)
            else:
                optimal_candidates_callback_ = None
            if suboptimal_candidates_callback is not None:
                suboptimal_candidates_callback_ \
                    = lambda s, n, t: suboptimal_candidates_callback(s, n, t, None)
            else:
                suboptimal_candidates_callback_ = None
            self._inner = RandomizedSearch(
                distr,
                score_fn_,
                greater_score_fn,
                weak_greater_score_fn,
                top_k_kept,
                n_jobs,
                sample_callback,
                optimal_candidates_callback_,
                suboptimal_candidates_callback_,
                top_k_callback,
            )
        else:
            if optimal_candidates_callback is not None:
                optimal_candidates_callback_ \
                    = lambda s, o, c: optimal_candidates_callback(s, o, None, c)
            else:
                optimal_candidates_callback_ = None
            if suboptimal_candidates_callback is not None:
                suboptimal_candidates_callback_ \
                    = lambda s, n, c: suboptimal_candidates_callback(s, n, None, c)
            else:
                suboptimal_candidates_callback_ = None
            self._inner = ExhaustiveSearch(
                distr,
                score_fn_,
                greater_score_fn,
                weak_greater_score_fn,
                top_k_kept,
                n_jobs,
                sample_callback,
                optimal_candidates_callback_,
                suboptimal_candidates_callback_,
                top_k_callback,
            )

    def search(self, n_iter: int) -> bool:
        return self._inner.search(n_iter)

    def optimal_candidates(self) -> list[Sample[UpdateStrategy, S]]:
        return self._inner.survivors()

    def top_k(self) -> list[Sample[UpdateStrategy, S]]:
        return self._inner.top_k()

    def __enter__(self) -> BaseUpdateStrategySearch[S]:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        if self._pool is not None:
            self._pool.stop()
        self._model_clone_helper.stop()


class UpdateStrategySearch(BaseUpdateStrategySearch[list[Any]]):

    def __init__(
        self,
        search_space: Any,
        model: Any,
        x_arr: Iterable[Any],
        y_arr: Iterable[Any],
        t_x_arr: Iterable[Any],
        t_y_arr: Iterable[Any],
        metric: Metric | Sequence[Metric | Literal['time', 'n_updates', 'n_samples']],
        optim_mode: Literal['min', 'max'] | Sequence[Literal['min', 'max']] | None,
        *,
        weight_fn: Callable[..., Any] | None = None,
        top_k_kept: int = 0,
        exhaustive: bool = False,
        n_jobs: int | None = None,
        resource_pool: BaseResourcePool[Any] | None = None,
        verbose_level: Literal[0, 1, 2] = 2,
        trace_history_file_path: str | PathLike | None = None,
        exhaustive_count_file_path: str | PathLike | None = None,
        optimal_scores_csv_path: str | PathLike | None = None,
        optimal_samples_file_path: str | PathLike | None = None,
        top_k_scores_csv_path: str | PathLike | None = None,
        top_k_samples_file_path: str | PathLike | None = None,
        load_old_state: bool = False,
        stop_signal_file_path: str | PathLike | None = None,
        eval_retry: int = 0,
        debug: bool = False,
        sample_callback: Callable[[Sample[UpdateStrategy, None]], None] | None = None,
        optimal_candidates_callback: Callable[
            [
                Sample[UpdateStrategy, list[Any]],
                list[Sample[UpdateStrategy, list[Any]]],
                set[Hashable] | None,
                int | None,
            ],
            bool,
        ] | None = None,
        suboptimal_candidates_callback: Callable[
            [Sample[UpdateStrategy, list[Any]], bool, set[Hashable] | None, int | None],
            bool,
        ] | None = None,
        top_k_callback: Callable[
            [Sample[UpdateStrategy, list[Any]], list[Sample[UpdateStrategy, list[Any]]]],
            None,
        ] | None = None,
    ) -> None:

        metric_: list[Metric | Literal['time', 'n_updates', 'n_samples']]
        match metric:
            case Metric():
                metric_ = [metric]
                metric__ = [metric]
            case Sequence():
                metric_ = list(metric)
                metric__ = [m for m in metric_ if isinstance(m, Metric)]
        del metric

        optim_mode_: list[Literal['min', 'max']] | None
        match optim_mode:
            case None:
                optim_mode_ = None
            case 'min':
                optim_mode_ = ['min']
            case 'max':
                optim_mode_ = ['max']
            case _:
                optim_mode_ = list(optim_mode)
        del optim_mode

        def score_fn(score: list[float], stats: Stats) -> list[Any]:
            score_and_stats = list[float | int]()
            score_i = 0
            for m in metric_:
                match m:
                    case Metric():
                        score_and_stats.append(score[score_i])
                        score_i += 1
                    case 'time':
                        score_and_stats.append(stats.model_update_cost_by_time())
                    case 'n_updates':
                        score_and_stats.append(stats.model_update_cost_by_num_updates())
                    case 'n_samples':
                        score_and_stats.append(stats.model_update_cost_by_num_samples())
            if weight_fn is None:
                return score_and_stats
            weighted = weight_fn(*score_and_stats)
            if isinstance(weighted, Sequence):
                return list(weighted)
            return [weighted]

        if optim_mode_ is not None:
            def greater_score_fn(s1: list[Any], s2: list[Any]) -> bool:
                if not (len(s1) == len(s2) == len(optim_mode_)):
                    raise RuntimeError
                greater = False
                for s1_, s2_, o in zip(s1, s2, optim_mode_, strict=True):
                    match o:
                        case 'min':
                            if s1_ > s2_:
                                return False
                            if s1_ < s2_:
                                greater = True
                        case 'max':
                            if s1_ < s2_:
                                return False
                            if s1_ > s2_:
                                greater = True
                return greater
            greater_score_fn_ = greater_score_fn
        else:
            greater_score_fn_ = None

        if optim_mode_ is not None:
            def weak_greater_score_fn(s1: list[Any], s2: list[Any]) -> bool:
                if not (len(s1) == len(s2) == len(optim_mode_)):
                    raise RuntimeError
                for s1_, s2_, o in zip(s1, s2, optim_mode_, strict=True):
                    match o:
                        case 'min':
                            if s1_ < s2_:
                                return True
                            if s1_ > s2_:
                                return False
                        case 'max':
                            if s1_ > s2_:
                                return True
                            if s1_ < s2_:
                                return False
                return False
            weak_greater_score_fn_ = weak_greater_score_fn
        else:
            weak_greater_score_fn_ = None

        multiprocessing_manager = Manager()
        log_queue = multiprocessing_manager.Queue()
        log_thread = Thread(target=self._log_thread_target, args=(log_queue,), daemon=True)
        log_thread.start()

        if sample_callback is None:
            def sample_callback_(sample: Sample[UpdateStrategy, None]) -> None:
                if verbose_level >= 1:
                    log_queue.put({'event': 'evaluating', 'sample': sample.explain()})
            sample_callback__ = sample_callback_
        else:
            sample_callback__ = sample_callback

        if optimal_candidates_callback is None:
            def optimal_candidates_callback_(
                sample: Sample[UpdateStrategy, list[Any]],
                optimal_candidates_now: list[Sample[UpdateStrategy, list[Any]]],
                trace_history_now: set[Hashable] | None,
                exhaustive_count_now: int | None,
            ) -> bool:
                self._log_event(
                    sample,
                    optimal_candidates_now,
                    verbose_level,
                    'optimal',
                    'optimal_candidates_now',
                    log_queue,
                )
                if trace_history_file_path is not None:
                    if trace_history_now is not None:
                        self._write_traces(trace_history_now, trace_history_file_path)
                if exhaustive_count_file_path is not None:
                    if exhaustive_count_now is not None:
                        self._write_exhaustive_count(
                            exhaustive_count_now,
                            exhaustive_count_file_path,
                        )
                if optimal_scores_csv_path is not None:
                    self._write_scores(
                        optimal_candidates_now,
                        optimal_scores_csv_path,
                        metric_,
                        weight_fn,
                    )
                if optimal_samples_file_path is not None:
                    self._write_samples(optimal_candidates_now, optimal_samples_file_path)
                if stop_signal_file_path is not None and Path(stop_signal_file_path).is_file():
                    Path(stop_signal_file_path).unlink()
                    return True
                return False
            optimal_candidates_callback__ = optimal_candidates_callback_
        else:
            optimal_candidates_callback__ = optimal_candidates_callback

        if suboptimal_candidates_callback is None:
            def suboptimal_candidates_callback_(
                sample: Sample[UpdateStrategy, list[Any]],
                is_new: bool,
                trace_history_now: set[Hashable] | None,
                exhaustive_count_now: int | None,
            ) -> bool:
                if verbose_level >= 1:
                    log_queue.put({
                        'event': 'suboptimal',
                        'sample': sample.explain(),
                        'is_new': is_new,
                        'score': sample.score if len(sample.score) != 1 else sample.score[0],
                    })
                if is_new:
                    if trace_history_file_path is not None:
                        if trace_history_now is not None:
                            self._write_traces(trace_history_now, trace_history_file_path)
                    if exhaustive_count_file_path is not None:
                        if exhaustive_count_now is not None:
                            self._write_exhaustive_count(
                                exhaustive_count_now,
                                exhaustive_count_file_path,
                            )
                    if stop_signal_file_path is not None and Path(stop_signal_file_path).is_file():
                        Path(stop_signal_file_path).unlink()
                        return True
                return False
            suboptimal_candidates_callback__ = suboptimal_candidates_callback_
        else:
            suboptimal_candidates_callback__ = suboptimal_candidates_callback

        if top_k_callback is None:
            def top_k_callback_(
                sample: Sample[UpdateStrategy, list[Any]],
                top_k_now: list[Sample[UpdateStrategy, list[Any]]],
            ) -> None:
                self._log_event(
                    sample,
                    top_k_now,
                    verbose_level,
                    'top_k',
                    'top_k_now',
                    log_queue,
                )
                if top_k_scores_csv_path is not None:
                    self._write_scores(top_k_now, top_k_scores_csv_path, metric_, weight_fn)
                if top_k_samples_file_path is not None:
                    self._write_samples(top_k_now, top_k_samples_file_path)
            top_k_callback__ = top_k_callback_
        else:
            top_k_callback__ = top_k_callback

        super().__init__(
            into_distr(search_space),
            apply_model_adaptors(model),
            x_arr,
            y_arr,
            t_x_arr,
            t_y_arr,
            metric__,
            score_fn,
            greater_score_fn_,
            weak_greater_score_fn_,
            eval_retry,
            top_k_kept,
            exhaustive,
            n_jobs,
            resource_pool,
            sample_callback__,
            optimal_candidates_callback__,
            suboptimal_candidates_callback__,
            top_k_callback__,
            debug,
        )
        self._multiprocessing_manager = multiprocessing_manager
        self._log_queue = log_queue
        self._log_thread = log_thread

        if load_old_state:
            self._load_old_state(
                exhaustive,
                trace_history_file_path,
                exhaustive_count_file_path,
                optimal_scores_csv_path,
                optimal_samples_file_path,
                top_k_scores_csv_path,
                top_k_samples_file_path,
            )

    @staticmethod
    def _log_thread_target(log_queue: Queue[Any]) -> None:
        while (info := log_queue.get()) is not None:
            logging.info(info)

    @staticmethod
    def _log_event(
        cur_sample: Sample[UpdateStrategy, list[Any]],
        all_samples: list[Sample[UpdateStrategy, list[Any]]],
        verbose_level: Literal[0, 1, 2],
        event_name: str,
        all_samples_name: str,
        log_queue: Queue[Any],
    ) -> None:
        if verbose_level >= 1:
            info = {
                'event': event_name,
                'sample': cur_sample.explain(),
                'score': cur_sample.score if len(cur_sample.score) != 1 else cur_sample.score[0],
            }
            if verbose_level >= 2:
                info[all_samples_name] = [
                    {
                        'sample': s.explain(),
                        'score': s.score if len(s.score) != 1 else s.score[0],
                    }
                    for s in all_samples
                ]
            log_queue.put(info)

    @staticmethod
    def _write_traces_unsafe(traces: set[Hashable], file_path: str | PathLike) -> None:
        with open(file_path, 'w') as f:
            for t in traces:
                print(repr(t), file=f)

    @staticmethod
    def _write_traces(traces: set[Hashable], file_path: str | PathLike) -> None:
        return safe_save(lambda p: UpdateStrategySearch._write_traces_unsafe(traces, p), file_path)

    @staticmethod
    def _write_exhaustive_count_unsafe(exhaustive_count: int, file_path: str | PathLike) -> None:
        Path(file_path).write_text(str(exhaustive_count))

    @staticmethod
    def _write_exhaustive_count(exhaustive_count: int, file_path: str | PathLike) -> None:
        return safe_save(
            lambda p: UpdateStrategySearch._write_exhaustive_count_unsafe(exhaustive_count, p),
            file_path,
        )

    @staticmethod
    def _write_scores(
        samples: list[Sample[UpdateStrategy, list[Any]]],
        csv_path: str | PathLike,
        metrics: list[Metric | Literal['time', 'n_updates', 'n_samples']],
        weight_fn: Callable[..., Any] | None,
    ) -> None:
        if len(samples) == 0:
            raise ValueError
        scores = [s.score for s in samples]
        if weight_fn is None:
            scores_columns = list[str]()
            metric_i = 0
            for m in metrics:
                match m:
                    case Metric():
                        scores_columns.append(f'metric_{metric_i}')
                        metric_i += 1
                    case _:
                        scores_columns.append(m)
        else:
            scores_columns = [f'weighted_{i}' for i in range(len(scores[0]))]
        safe_save(
            pd.DataFrame(scores, columns=scores_columns).rename_axis(index='i').to_csv,
            csv_path,
        )

    @staticmethod
    def _write_samples_unsafe(
        samples: list[Sample[UpdateStrategy, list[Any]]],
        file_path: str | PathLike,
    ) -> None:
        with open(file_path, 'w') as f:
            for i, s in enumerate(samples):
                i = str(i).rjust(len(str(len(samples) - 1)))
                print(f'{i}: {s.explain()} ##BEGIN#TRACE##{repr(s.trace)}##END#TRACE##', file=f)

    @staticmethod
    def _write_samples(
        samples: list[Sample[UpdateStrategy, list[Any]]],
        file_path: str | PathLike,
    ) -> None:
        return safe_save(
            lambda p: UpdateStrategySearch._write_samples_unsafe(samples, p),
            file_path,
        )

    def _load_old_state(
        self,
        exhaustive: bool,
        trace_history_file_path: str | PathLike | None,
        exhaustive_count_file_path: str | PathLike | None,
        optimal_scores_csv_path: str | PathLike | None,
        optimal_samples_file_path: str | PathLike | None,
        top_k_scores_csv_path: str | PathLike | None,
        top_k_samples_file_path: str | PathLike | None,
    ) -> None:

        if optimal_scores_csv_path is None:
            raise ValueError
        if optimal_samples_file_path is None:
            raise ValueError
        if top_k_scores_csv_path is None:
            raise ValueError
        if top_k_samples_file_path is None:
            raise ValueError
        if not exhaustive and trace_history_file_path is None:
            raise ValueError
        if exhaustive and exhaustive_count_file_path is None:
            raise ValueError

        if all(map(self._check_non_existence, [
            trace_history_file_path if not exhaustive else exhaustive_count_file_path,
            optimal_scores_csv_path,
            optimal_samples_file_path,
            top_k_scores_csv_path,
            top_k_samples_file_path,
        ])):
            return

        optimal_scores_csv_path = self._check_existence(optimal_scores_csv_path)
        optimal_samples_file_path = self._check_existence(optimal_samples_file_path)
        top_k_scores_csv_path = self._check_existence(top_k_scores_csv_path)
        top_k_samples_file_path = self._check_existence(top_k_samples_file_path)

        optimal_candidates = zip(
            self._load_traces_from_samples_file(optimal_samples_file_path),
            self._load_scores_from_csv(optimal_scores_csv_path),
            strict=True,
        )
        top_k = zip(
            self._load_traces_from_samples_file(top_k_samples_file_path),
            self._load_scores_from_csv(top_k_scores_csv_path),
            strict=True,
        )

        if not exhaustive:
            trace_history_file_path = self._check_existence(trace_history_file_path)
            trace_history = {
                eval(l)
                for l in Path(trace_history_file_path).read_text().splitlines()
            }
            typing.cast(RandomizedSearch[UpdateStrategy, list[Any]], self._inner).load_state(
                optimal_candidates,
                top_k,
                trace_history,
            )
        else:
            exhaustive_count_file_path = self._check_existence(exhaustive_count_file_path)
            exhaustive_count = eval(Path(exhaustive_count_file_path).read_text())
            typing.cast(ExhaustiveSearch[UpdateStrategy, list[Any]], self._inner).load_state(
                optimal_candidates,
                top_k,
                exhaustive_count,
            )

    @staticmethod
    def _check_non_existence(path: str | PathLike | None) -> bool:
        return path is not None and not Path(path).exists()

    @staticmethod
    def _check_existence(path: str | PathLike | None) -> str | PathLike:
        if path is None:
            raise ValueError
        if not Path(path).is_file():
            raise FileNotFoundError
        return path

    @staticmethod
    def _load_traces_from_samples_file(path: str | PathLike) -> list[Hashable]:
        traces = []
        for line in Path(path).read_text().splitlines():
            match = re.search('##BEGIN#TRACE##(.*)##END#TRACE##', line)
            if match is None:
                raise RuntimeError
            traces.append(eval(match.group(1)))
        return traces

    @staticmethod
    def _load_scores_from_csv(path: str | PathLike) -> list[list[Any]]:
        df = pd.read_csv(path)
        cols = [df[c].astype(object).values for c in df.columns[1:]]
        return [list(row) for row in zip(*cols, strict=True)]

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._log_queue.put(None)
        self._log_thread.join()
        self._multiprocessing_manager.shutdown()
        return super().__exit__(exc_type, exc_value, traceback)


__all__ = [
    'UpdateStrategySearch',
]
