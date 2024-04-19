from collections.abc import Callable
from io import StringIO
from typing import Any, Generic, Hashable, TypeVar

from musc.commons import explain_object
from musc.hpo.distr.a import (
    BaseDistribution,
    DistributionChoice,
    DistributionJoint,
    DistributionOnePoint,
    DistributionScipyWrapper,
)


V = TypeVar('V')


def into_distr(obj: Any) -> BaseDistribution[Any]:
    if isinstance(obj, BaseDistribution):
        return obj
    if hasattr(obj, 'rvs'):
        return DistributionScipyWrapper(obj)
    if isinstance(obj, list):
        return DistributionChoice([into_distr(obj_) for obj_ in obj])
    if isinstance(obj, dict):
        obj = dict(obj)
        if 'type' in obj:
            return DistributionJointType(
                obj.pop('type'),
                {k: into_distr(v) for k, v in obj.items()},
            )
        if 'base_fn' in obj:
            return DistributionJointBaseFn(
                obj.pop('base_fn'),
                {k: into_distr(v) for k, v in obj.items()},
            )
        raise ValueError
    return DistributionOnePoint(obj)


class DistributionJointType(Generic[V], DistributionJoint[V]):

    def __init__(
        self,
        type_: Callable[..., V],
        kwargs: dict[str, BaseDistribution[Any]],
    ) -> None:
        super().__init__(type_, kwargs=kwargs)


class DistributionJointBaseFn(DistributionJoint[Any]):

    def __init__(
        self,
        base_fn: Callable[..., Any],
        kwargs: dict[str, BaseDistribution[Any]],
    ):
        super().__init__(lambda **ka_: lambda *a, **ka: base_fn(*a, **ka, **ka_), kwargs=kwargs)
        self._base_fn = base_fn

    def explain_trace(self, trace: tuple[Hashable, ...]) -> str:
        expr = StringIO()
        expr.write(f'(lambda *args, **kwargs: {explain_object(self._base_fn)}(*args, **kwargs')
        for (k, d), st in zip(self._kwargs.items(), trace[len(self._args):]):
            expr.write(f', {k}={d.explain_trace(st)}')
        expr.write('))')
        return expr.getvalue()


class SingleValue(Generic[V], DistributionOnePoint[V]):

    pass


__all__ = [
    'into_distr',
    'SingleValue',
]
