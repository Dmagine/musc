from __future__ import annotations

import functools
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from joblib import Parallel
from readerwriterlock.rwlock import RWLockFair
from sortedcontainers import SortedList as SortedListBase

from musc.commons import joblib_imap_unordered
from musc.hpo.distr.a import BaseDistribution


V, S = TypeVar('V'), TypeVar('S')
V_, S_ = TypeVar('V_'), TypeVar('S_')
T = TypeVar('T')


class BaseRandomizedSearch(Generic[V, S], ABC):

    class Portable(Generic[V_, S_]):

        def __init__(
            self,
            distr: BaseDistribution[V_],
            score_fn: Callable[[V_], S_],
            sample_callback: Callable[[Sample[V_, None]], None] | None = None,
        ) -> None:

            self._distr = distr
            self._score_fn = score_fn
            self._sample_callback = sample_callback

        def sample_by_trace(self, trace: Hashable) -> Sample[V_, S_]:
            value = self._distr.sample_by_trace(trace)
            if self._sample_callback is not None:
                self._sample_callback(Sample(self._distr, trace, value, None))
            score = self._score_fn(value)
            return Sample(self._distr, trace, value, score)

    def __init__(
        self,
        distr: BaseDistribution[V],
        score_fn: Callable[[V], S],
        greater_score_fn: Callable[[S, S], bool] | None,
        weak_greater_score_fn: Callable[[S, S], bool] | None = None,
        top_k_kept: int = 0,
        n_jobs: int | None = None,
        sample_callback: Callable[[Sample[V, None]], None] | None = None,
        survivors_callback: Callable[[Sample[V, S], list[Sample[V, S]]], bool] | None = None,
        outsider_callback: Callable[[Sample[V, S], bool], bool] | None = None,
        top_k_callback: Callable[[Sample[V, S], list[Sample[V, S]]], None] | None = None,
    ) -> None:

        if top_k_kept >= 1 and weak_greater_score_fn is None:
            raise ValueError

        self._portable = BaseRandomizedSearch.Portable(distr, score_fn, sample_callback)
        self._greater_score_fn = greater_score_fn
        self._weak_greater_score_fn = weak_greater_score_fn
        self._n_jobs = n_jobs
        self._survivors_callback = survivors_callback
        self._outsider_callback = outsider_callback
        self._top_k_callback = top_k_callback

        self._survivors = list[Sample[V, S]]()
        self._top_k = SortedList(
            key=functools.cmp_to_key(self._sample_weak_cmp),
            capacity=top_k_kept,
        )

        # self._lock cannot be sent to joblib process pool.
        self._lock = RWLockFair()

    def load_state(
        self,
        survivors: Iterable[tuple[Hashable, S]],
        top_k: Iterable[tuple[Hashable, S]],
    ) -> None:
        self._survivors.clear()
        self._top_k.clear()
        for trace, score in survivors:
            self._survivors.append(Sample(
                self._portable._distr,
                trace,
                self._portable._distr.sample_by_trace(trace),
                score,
            ))
        for trace, score in top_k:
            self._top_k.add(Sample(
                self._portable._distr,
                trace,
                self._portable._distr.sample_by_trace(trace),
                score,
            ))

    @abstractmethod
    def random_new_traces(self, n: int) -> list[Hashable]:
        pass

    def search(self, n_iter: int) -> bool:
        traces = self.random_new_traces(n_iter)
        parallel = Parallel(n_jobs=self._n_jobs, return_as='generator')
        for sample in joblib_imap_unordered(parallel, self._portable.sample_by_trace, traces):
            survived = True
            with self._lock.gen_wlock():
                to_kill = list[tuple[int, Sample[V, S]]]()
                if self._greater_score_fn is not None:
                    for i, sample_ in enumerate(self._survivors):
                        if self._greater_score_fn(sample_.score, sample.score):
                            survived = False
                            break
                        if self._greater_score_fn(sample.score, sample_.score):
                            to_kill.append((i, sample))
                if survived:
                    for i, _ in reversed(to_kill):
                        del self._survivors[i]
                    self._survivors.append(sample)
                in_top_k = self._top_k.add(sample)
            should_stop = False
            if survived and self._survivors_callback is not None:
                should_stop = self._survivors_callback(sample, self.survivors())
                if self._outsider_callback is not None:
                    for _, s in to_kill:
                        self._outsider_callback(s, False)
            if not survived and self._outsider_callback is not None:
                should_stop = self._outsider_callback(sample, True)
            if in_top_k and self._top_k_callback is not None:
                self._top_k_callback(sample, self.top_k())
            if should_stop:
                return False
        return True

    def _sample_weak_cmp(self, s1: Sample[V, S], s2: Sample[V, S]) -> int:
        if self._weak_greater_score_fn is None:
            return 0
        if self._weak_greater_score_fn(s1.score, s2.score):
            return -1
        if self._weak_greater_score_fn(s2.score, s1.score):
            return 1
        return 0

    def survivors(self) -> list[Sample[V, S]]:
        with self._lock.gen_rlock():
            if self._weak_greater_score_fn is not None:
                return sorted(self._survivors, key=functools.cmp_to_key(self._sample_weak_cmp))
            return list(self._survivors)

    def top_k(self) -> list[Sample[V, S]]:
        with self._lock.gen_rlock():
            return self._top_k.data()


class RandomizedSearch(Generic[V, S], BaseRandomizedSearch[V, S]):

    def __init__(
        self,
        distr: BaseDistribution[V],
        score_fn: Callable[[V], S],
        greater_score_fn: Callable[[S, S], bool] | None,
        weak_greater_score_fn: Callable[[S, S], bool] | None = None,
        top_k_kept: int = 0,
        n_jobs: int | None = None,
        sample_callback: Callable[[Sample[V, None]], None] | None = None,
        survivors_callback: \
            Callable[[Sample[V, S], list[Sample[V, S]], set[Hashable]], bool] | None = None,
        outsider_callback: Callable[[Sample[V, S], bool, set[Hashable]], bool] | None = None,
        top_k_callback: Callable[[Sample[V, S], list[Sample[V, S]]], None] | None = None,
    ) -> None:
        def survivors_callback_(sample: Sample[V, S], survivors: list[Sample[V, S]]) -> bool:
            self._trace_history.add(sample.trace)
            if survivors_callback is not None:
                return survivors_callback(sample, survivors, self.trace_history())
            return False
        def outsider_callback_(sample: Sample[V, S], is_new: bool) -> bool:
            if is_new:
                self._trace_history.add(sample.trace)
            if outsider_callback is not None:
                return outsider_callback(sample, is_new, self.trace_history())
            return False
        super().__init__(
            distr,
            score_fn,
            greater_score_fn,
            weak_greater_score_fn,
            top_k_kept,
            n_jobs,
            sample_callback,
            survivors_callback_,
            outsider_callback_,
            top_k_callback,
        )
        self._trace_history = set[Hashable]()

    def load_state(
        self,
        survivors: Iterable[tuple[Hashable, S]],
        top_k: Iterable[tuple[Hashable, S]],
        trace_history: set[Hashable],
    ) -> None:
        BaseRandomizedSearch.load_state(self, survivors, top_k)
        self._trace_history.clear()
        for trace in trace_history:
            self._trace_history.add(trace)

    def random_new_traces(self, n: int) -> list[Hashable]:
        traces = list[Hashable]()
        while True:
            if len(traces) == n:
                break
            if len(self._trace_history) + len(traces) == self._portable._distr.size():
                break
            trace = self._portable._distr.random_trace()
            if trace not in self._trace_history and trace not in traces:
                traces.append(trace)
        return traces

    def trace_history(self) -> set[Hashable]:
        with self._lock.gen_rlock():
            return set(self._trace_history)


class ExhaustiveSearch(Generic[V, S], BaseRandomizedSearch[V, S]):

    def __init__(
        self,
        distr: BaseDistribution[V],
        score_fn: Callable[[V], S],
        greater_score_fn: Callable[[S, S], bool] | None,
        weak_greater_score_fn: Callable[[S, S], bool] | None = None,
        top_k_kept: int = 0,
        n_jobs: int | None = None,
        sample_callback: Callable[[Sample[V, None]], None] | None = None,
        survivors_callback: Callable[[Sample[V, S], list[Sample[V, S]], int], bool] | None = None,
        outsider_callback: Callable[[Sample[V, S], bool, int], bool] | None = None,
        top_k_callback: Callable[[Sample[V, S], list[Sample[V, S]]], None] | None = None,
    ) -> None:
        if (d_s := self._portable._distr.size()) is None:
            raise ValueError
        def survivors_callback_(sample: Sample[V, S], survivors: list[Sample[V, S]]) -> bool:
            self._count += 1
            if survivors_callback is not None:
                return survivors_callback(sample, survivors, self.count())
            return False
        def outsider_callback_(sample: Sample[V, S], is_new: bool) -> bool:
            if is_new:
                self._count += 1
            if outsider_callback is not None:
                return outsider_callback(sample, is_new, self.count())
            return False
        super().__init__(
            distr,
            score_fn,
            greater_score_fn,
            weak_greater_score_fn,
            top_k_kept,
            n_jobs,
            sample_callback,
            survivors_callback_,
            outsider_callback_,
            top_k_callback,
        )
        self._d_s = d_s
        self._count = 0

    def load_state(
        self,
        survivors: Iterable[tuple[Hashable, S]],
        top_k: Iterable[tuple[Hashable, S]],
        count: int,
    ) -> None:
        BaseRandomizedSearch.load_state(self, survivors, top_k)
        self._count = count

    def random_new_traces(self, n: int) -> list[Hashable]:
        if self._count + n > self._d_s:
            raise IndexError
        return [self._portable._distr.trace_by_index(self._count + i) for i in range(n)]

    def count(self) -> int:
        with self._lock.gen_rlock():
            return self._count


@dataclass
class Sample(Generic[V, S]):

    distr: BaseDistribution[V]
    trace: Hashable
    value: V
    score: S

    def explain(self) -> str:
        return self.distr.explain_trace(self.trace)


class SortedList(Generic[T]):

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        key: Callable[[T], Any] | None = None,
        capacity: int = -1,
    ) -> None:
        self._inner = SortedListBase(iterable, key)
        self._key = key if key is not None else lambda _: 0
        self._capacity = capacity

    def data(self) -> list[T]:
        return list(self._inner)

    def add(self, value: T) -> bool:
        if self._capacity < 0 or len(self._inner) < self._capacity:
            self._inner.add(value)
            return True
        if len(self._inner) == 0 or self._key(value) >= self._key(typing.cast(T, self._inner[-1])):
            return False
        self._inner.add(value)
        self._inner.pop()
        return True

    def clear(self) -> None:
        self._inner.clear()


__all__ = [
    'RandomizedSearch',
    'ExhaustiveSearch',
    'Sample',
]
