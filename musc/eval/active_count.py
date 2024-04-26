from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Event, Lock


class BaseActiveCount(ABC):

    @abstractmethod
    def inc(self) -> None:
        pass

    @abstractmethod
    def dec(self) -> None:
        pass


class ActiveCount(BaseActiveCount):

    def __init__(self, count: int, idle_event: Event, lock: Lock) -> None:
        self._count = count
        self._idle_event = idle_event
        self._lock = lock

    @staticmethod
    def create() -> tuple[ActiveCount, Event]:
        idle_event = Event()
        idle_event.set()
        return ActiveCount(0, idle_event, Lock()), idle_event

    def inc(self) -> None:
        with self._lock:
            self._count += 1
            if self._count == 1:
                self._idle_event.clear()

    def dec(self) -> None:
        with self._lock:
            if self._count == 0:
                raise RuntimeError
            self._count -= 1
            if self._count == 0:
                self._idle_event.set()


class NullActiveCount(BaseActiveCount):

    def inc(self) -> None:
        pass

    def dec(self) -> None:
        pass


__all__ = [
    'BaseActiveCount',
    'ActiveCount',
    'NullActiveCount',
]
