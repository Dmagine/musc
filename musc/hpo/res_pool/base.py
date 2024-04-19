from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


R = TypeVar('R')


class BaseResourcePool(Generic[R], ABC):

    @abstractmethod
    def alloc(self) -> R | None:
        pass

    @abstractmethod
    def dealloc(self, resource: R) -> None:
        pass


__all__ = [
    'BaseResourcePool',
]
