from collections.abc import Callable, Generator, Iterable
from multiprocessing import Manager
from os import PathLike
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, TypeVar
from uuid import uuid4

import cloudpickle
from joblib import Parallel, delayed


R, T, V = TypeVar('R'), TypeVar('T'), TypeVar('V')


class _JoblibFinished:
    pass


def explain_object(obj: Any) -> str:
    if hasattr(obj, '__explain_self__') and not isinstance(obj, type):
        return obj.__explain_self__()
    if hasattr(obj, '__name__'):
        if hasattr(obj, '__module__') and obj.__module__ != '__main__':
            return f'{obj.__module__}.{obj.__name__}'
        return obj.__name__
    repr_obj = repr(obj)
    if repr_obj.startswith('<'):
        return f'{explain_object(type(obj))}(...)'
    return repr_obj


def joblib_imap_unordered(
    parallel: Parallel,
    func: Callable[[V], R],
    iter: Iterable[V],
) -> Generator[R, None, None]:
    with Manager() as multiprocessing_manager:
        queue: Queue[bytes | _JoblibFinished] = multiprocessing_manager.Queue()
        progress = parallel(map(delayed(lambda v: queue.put(cloudpickle.dumps(func(v)))), iter))
        monitor_thread = Thread(target=_joblib_progress_monitor, args=(queue, progress))
        monitor_thread.start()
        while not isinstance(value := queue.get(), _JoblibFinished):
            yield cloudpickle.loads(value)
        monitor_thread.join()


def safe_save(original_save: Callable[[str], T], path: str | PathLike) -> T:
    path = Path(path)
    tmp_path = str(uuid4()) + ''.join(path.suffixes)
    return_value = original_save(tmp_path)
    path.unlink(missing_ok=True)
    Path(tmp_path).rename(path)
    return return_value


def _joblib_progress_monitor(queue: Queue[Any], progress: Iterable[Any]) -> Any:
    for _ in progress:
        pass
    queue.put(_JoblibFinished())


__all__ = [
    'explain_object',
    'joblib_imap_unordered',
    'safe_save',
]
