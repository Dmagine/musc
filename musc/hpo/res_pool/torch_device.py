from collections import Counter
from collections.abc import Sequence

import torch

from musc.hpo.res_pool.base import BaseResourcePool


class BaseTorchDevicePool(BaseResourcePool[torch.device]):

    def __init__(self, indices: Sequence[int], type_: str) -> None:
        self._index_counter = Counter(indices)
        self._type = type_

    def alloc(self) -> torch.device | None:
        if (index := next(iter(self._index_counter.keys()), None)) is None:
            return None
        self._index_counter.subtract([index])
        self._index_counter = +self._index_counter
        return torch.device(self._type, index)

    def dealloc(self, resource: torch.device) -> None:
        if resource.type != self._type:
            raise ValueError
        self._index_counter.update([resource.index])


class TorchDevicePool(BaseTorchDevicePool):

    def __init__(self, indices: Sequence[int], repeat: int = 1, type_: str = 'cuda') -> None:
        super().__init__(list(indices) * repeat, type_)


__all__ = [
    'TorchDevicePool',
]
