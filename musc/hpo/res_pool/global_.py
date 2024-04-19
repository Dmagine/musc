from __future__ import annotations

import time
import typing
from multiprocessing import Manager, Pipe, Process
from multiprocessing.connection import _ConnectionBase
from queue import Queue
from typing import Generic, Literal, TypeVar

from musc.hpo.res_pool.base import BaseResourcePool


R = TypeVar('R')


QueueToDaemon = Queue[tuple[Literal['alloc'], _ConnectionBase] | tuple[Literal['dealloc'], R] | None]


class GlobalResourcePool(Generic[R]):

    def __init__(self, inner: BaseResourcePool[R]) -> None:
        self._multiprocessing_manager = Manager()
        self._queue_to_daemon: QueueToDaemon[R] = self._multiprocessing_manager.Queue()
        # self._process cannot be sent to joblib process pool.
        self._process = Process(
            target=process_target,
            args=(self._queue_to_daemon, inner),
            daemon=True,
        )
        self._process.start()

    def portable(self) -> GlobalResourcePoolPortable[R]:
        return GlobalResourcePoolPortable(self._queue_to_daemon)

    def stop(self) -> None:
        self._queue_to_daemon.put(None)
        self._process.join()
        self._multiprocessing_manager.shutdown()


class GlobalResourcePoolPortable(Generic[R]):

    def __init__(self, queue_to_daemon: QueueToDaemon[R]) -> None:
        self._queue_to_daemon = queue_to_daemon

    def alloc(self) -> ResourceContext[R]:
        return ResourceContext(self, self.alloc_manual())

    def alloc_manual(self) -> R:
        while True:
            resp_receiver, resp_sender = Pipe(duplex=False)
            self._queue_to_daemon.put(('alloc', resp_sender))
            if (resource := typing.cast(R | None, resp_receiver.recv())) is not None:
                return resource
            time.sleep(1.0)

    def dealloc_manual(self, resource: R) -> None:
        self._queue_to_daemon.put(('dealloc', resource))


class ResourceContext(Generic[R]):

    def __init__(self, pool: GlobalResourcePoolPortable[R], resource: R) -> None:
        self._pool = pool
        self._resource = resource

    def __enter__(self) -> R:
        return self._resource

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self._pool.dealloc_manual(self._resource)


def process_target(queue: QueueToDaemon[R], pool: BaseResourcePool[R]) -> None:
    while True:
        match queue.get():
            case ('alloc', resp_sender):
                resp_sender.send(pool.alloc())
            case ('dealloc', resource):
                pool.dealloc(resource)
            case None:
                break


__all__ = [
    'GlobalResourcePool',
]
