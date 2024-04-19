from __future__ import annotations

import multiprocessing
from multiprocessing import Manager, Pipe
from multiprocessing.connection import _ConnectionBase
from queue import Queue
from typing import Any, Generic, TypeVar

from musc.service.concepts.a import BaseModel
from musc.service.concepts.high_level.a import ModelGuard


XR, X, YR, Y, YP = TypeVar('XR'), TypeVar('X'), TypeVar('YR'), TypeVar('Y'), TypeVar('YP')


QueueToDaemon = Queue[_ConnectionBase | None]


class ModelCloneHelper(Generic[XR, X, YR, Y, YP]):

    def __init__(self, model: BaseModel[XR, X, YR, Y, YP]) -> None:
        self.__multiprocessing_manager = Manager()
        self.__queue_to_daemon: QueueToDaemon = self.__multiprocessing_manager.Queue()
        # self._process cannot be sent to joblib process pool.
        # Use 'spawn' here. See https://github.com/pytorch/pytorch/issues/122186.
        self.__process = multiprocessing.get_context('spawn').Process(
            target=process_target,
            args=(self.__queue_to_daemon, model),
            daemon=True,
        )
        self.__process.start()

    def portable(self) -> ModelCloneHelperPortable[XR, X, YR, Y, YP]:
        return ModelCloneHelperPortable(self.__queue_to_daemon)

    def stop(self) -> None:
        self.__queue_to_daemon.put(None)
        self.__process.join()
        self.__multiprocessing_manager.shutdown()


class ModelCloneHelperPortable(Generic[XR, X, YR, Y, YP], ModelGuard[XR, X, YR, Y, YP]):

    def __init__(self, queue_to_daemon: QueueToDaemon) -> None:
        self.__queue_to_daemon = queue_to_daemon

    def __get(self) -> BaseModel[XR, X, YR, Y, YP]:
        resp_receiver, resp_sender = Pipe(duplex=False)
        self.__queue_to_daemon.put(resp_sender)
        return resp_receiver.recv()

    def clone(self) -> BaseModel[XR, X, YR, Y, YP]:
        return self.__get().clone()

    def clone_with_resource(self, resource: Any) -> BaseModel[XR, X, YR, Y, YP]:
        return self.__get().clone_with_resource(resource)


def process_target(queue: QueueToDaemon, model: BaseModel[XR, X, YR, Y, YP]) -> None:
    while True:
        match queue.get():
            case None:
                break
            case resp_sender:
                resp_sender.send(model)


__all__ = [
    'ModelCloneHelper',
]
