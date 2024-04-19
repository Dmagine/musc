from __future__ import annotations

import itertools
import random
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from io import StringIO
from typing import Any, Generic, Protocol, TypeVar

from musc.commons import explain_object


V, VH = TypeVar('V', covariant=True), TypeVar('VH', bound=Hashable)


class BaseDistribution(Generic[V], ABC):

    @abstractmethod
    def size(self) -> int | None:
        pass

    @abstractmethod
    def random_trace(self) -> Hashable:
        pass

    @abstractmethod
    def trace_by_index(self, index: int) -> Hashable:
        pass

    @abstractmethod
    def sample_by_trace(self, trace: Hashable) -> V:
        pass

    @abstractmethod
    def explain_trace(self, trace: Hashable) -> str:
        pass


class DistributionOnePoint(Generic[V], BaseDistribution[V]):

    def __init__(self, v: V) -> None:
        self._v = v

    def size(self) -> int | None:
        return 1

    def random_trace(self) -> None:
        return None

    def trace_by_index(self, index: int) -> None:
        if index != 0:
            raise IndexError
        return None

    def sample_by_trace(self, trace: None) -> V:
        del trace
        return self._v

    def explain_trace(self, trace: None) -> str:
        del trace
        return explain_object(self._v)


class DistributionChoice(Generic[V], BaseDistribution[V]):

    def __init__(
        self,
        population: list[BaseDistribution[V]],
        weights: list[float] | None = None,
    ) -> None:

        if len(population) == 0:
            raise ValueError

        self._population = population
        self._weights = weights

        self._size = 0
        self._cum_size = [0] * len(self._population)
        for i, d in enumerate(self._population):
            if (d_s := d.size()) is None:
                self._size = None
                self._cum_size = None
                break
            self._size += d_s
            for j in range(i+1, len(self._population)):
                self._cum_size[j] += d_s

    def size(self) -> int | None:
        return self._size

    def random_trace(self) -> tuple[int, Hashable]:
        index_in_population = random.choices(range(len(self._population)), self._weights)[0]
        return (index_in_population, self._population[index_in_population].random_trace())

    def trace_by_index(self, index: int) -> tuple[int, Hashable]:
        if self._size is None or self._cum_size is None or index >= self._size or index < 0:
            raise IndexError
        for i in range(len(self._population)-1, -1, -1):
            if index >= self._cum_size[i]:
                return (i, self._population[i].trace_by_index(index - self._cum_size[i]))
        assert False

    def sample_by_trace(self, trace: tuple[int, Hashable]) -> V:
        if trace[0] >= len(self._population) or trace[0] < 0:
            raise ValueError
        return self._population[trace[0]].sample_by_trace(trace[1])

    def explain_trace(self, trace: tuple[int, Hashable]) -> str:
        if trace[0] >= len(self._population) or trace[0] < 0:
            raise ValueError
        return self._population[trace[0]].explain_trace(trace[1])


class DistributionJoint(Generic[V], BaseDistribution[V]):

    def __init__(
        self,
        callable: Callable[..., V],
        args: list[BaseDistribution[Any]] | None = None,
        kwargs: dict[str, BaseDistribution[Any]] | None = None,
        include_prelude: bool = True,
    ) -> None:

        self._callable = callable
        self._args = args if args is not None else []
        if include_prelude and hasattr(callable, 'kwargs_distr_prelude'):
            self._kwargs = typing.cast(
                dict[str, BaseDistribution[Any]],
                callable.kwargs_distr_prelude(),
            )
        else:
            self._kwargs = {}
        if kwargs is not None:
            for k, d in kwargs.items():
                self._kwargs[k] = d

        self._size = 1
        for d in itertools.chain(self._args, self._kwargs.values()):
            if (d_s := d.size()) is None:
                self._size = None
                break
            self._size *= d_s

    def size(self) -> int | None:
        return self._size

    def random_trace(self) -> tuple[Hashable, ...]:
        return tuple(d.random_trace() for d in itertools.chain(self._args, self._kwargs.values()))

    def trace_by_index(self, index: int) -> tuple[Hashable, ...]:
        if self._size is None or index >= self._size or index < 0:
            raise IndexError
        trace = []
        for d in itertools.chain(self._args, self._kwargs.values()):
            assert isinstance(d_s := d.size(), int)
            index, i = divmod(index, d_s)
            trace.append(d.trace_by_index(i))
        return tuple(trace)

    def sample_by_trace(self, trace: tuple[Hashable, ...]) -> V:
        if len(trace) != len(self._args) + len(self._kwargs):
            raise ValueError
        args = [d.sample_by_trace(st) for d, st in zip(self._args, trace[:len(self._args)])]
        kwargs = {
            k: d.sample_by_trace(st)
            for (k, d), st in zip(self._kwargs.items(), trace[len(self._args):])
        }
        return self._callable(*args, **kwargs)

    def explain_trace(self, trace: tuple[Hashable, ...]) -> str:
        if len(trace) != len(self._args) + len(self._kwargs):
            raise ValueError
        expr = StringIO()
        expr.write(explain_object(self._callable))
        expr.write('(')
        write_comma = False
        for d, st in zip(self._args, trace[:len(self._args)]):
            if not write_comma:
                write_comma = True
            else:
                expr.write(', ')
            expr.write(d.explain_trace(st))
        for (k, d), st in zip(self._kwargs.items(), trace[len(self._args):]):
            if not write_comma:
                write_comma = True
            else:
                expr.write(', ')
            expr.write(f'{k}={d.explain_trace(st)}')
        expr.write(')')
        return expr.getvalue()


class DistributionScipyWrapper(Generic[VH], BaseDistribution[VH]):

    def __init__(self, inner: ProtocolScipyDistr[VH]) -> None:
        self._inner = inner

    def size(self) -> int | None:
        return None

    def random_trace(self) -> VH:
        return self._inner.rvs()

    def trace_by_index(self, index: int) -> VH:
        del index
        raise IndexError

    def sample_by_trace(self, trace: VH) -> VH:
        return trace

    def explain_trace(self, trace: VH) -> str:
        return explain_object(trace)


class ProtocolScipyDistr(Generic[V], Protocol):

    def rvs(self) -> V:
        raise NotImplementedError


__all__ = [
    'BaseDistribution',
    'DistributionOnePoint',
    'DistributionChoice',
    'DistributionJoint',
    'DistributionScipyWrapper',
]
