from dataclasses import dataclass


class ServiceErrorByDuplicatedId(Exception):
    pass


@dataclass
class ServiceErrorFromModelPrediction(Exception):
    inner: BaseException


__all__ = [
    'ServiceErrorByDuplicatedId',
    'ServiceErrorFromModelPrediction',
]
