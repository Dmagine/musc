from __future__ import annotations

from dataclasses import dataclass

from readerwriterlock.rwlock import RWLockFair


class StatsWithLock:

    def __init__(self) -> None:
        self._inner = Stats([])
        self._lock = RWLockFair()

    def inner(self) -> Stats:
        return self._inner

    def gen_rlock(self) -> RWLockFair._aReader:
        return self._lock.gen_rlock()

    def gen_wlock(self) -> RWLockFair._aWriter:
        return self._lock.gen_wlock()


@dataclass
class Stats:
    model_update_records: list[ModelUpdateRecord]
    def model_update_cost_by_time(self) -> float:
        return sum(r.time_spent for r in self.model_update_records)
    def model_update_cost_by_num_updates(self) -> int:
        return len(self.model_update_records)
    def model_update_cost_by_num_samples(self) -> int:
        return sum(
            r.new_concept_range[1] - r.new_concept_range[0]
            for r in self.model_update_records
        )


@dataclass
class ModelUpdateRecord:
    drift_point: int
    new_concept_range: tuple[int, int]
    time_spent: float


__all__ = [
    'Stats',
    'StatsWithLock',
    'ModelUpdateRecord',
]
