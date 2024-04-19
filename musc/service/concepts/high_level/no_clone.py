from typing import Generic, TypeVar

from musc.service.concepts.a import BaseModel
from musc.service.concepts.high_level.a import ModelGuard


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


class ModelNoClone(Generic[XR, X, YR, Y, YP], ModelGuard[XR, X, YR, Y, YP]):

    def __init__(self, inner: BaseModel[XR, X, YR, Y, YP]) -> None:
        self.__inner = inner

    def clone(self) -> BaseModel[XR, X, YR, Y, YP]:
        return self.__inner


def no_clone(model: BaseModel[XR, X, YR, Y, YP]) -> ModelNoClone[XR, X, YR, Y, YP]:
    return ModelNoClone(model)


__all__ = [
    'no_clone',
]
