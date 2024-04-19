from typing import Any

import river.drift

from musc.hpo.distr.a import BaseDistribution
from musc.hpo.distr.high_level import into_distr


class ADWIN(river.drift.ADWIN):

    @staticmethod
    def kwargs_distr_prelude() -> dict[str, BaseDistribution[Any]]:
        return {
            'delta': into_distr([
                0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
            ]),
            'clock': into_distr([1]),
            'max_buckets': into_distr([4, 6, 8, 12, 16]),
            'min_window_length': into_distr([2, 3, 4, 6, 8, 12]),
            'grace_period': into_distr([2, 3, 4, 6, 8, 12, 16, 24]),
        }


class KSWIN(river.drift.KSWIN):

    @staticmethod
    def kwargs_distr_prelude() -> dict[str, BaseDistribution[Any]]:
        return {
            'alpha': into_distr([
                0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
                0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9,
            ]),
        }


class PageHinkley(river.drift.PageHinkley):

    @staticmethod
    def kwargs_distr_prelude() -> dict[str, BaseDistribution[Any]]:
        return {
            'min_instances': into_distr([2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]),
            'delta': into_distr([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]),
            'threshold': into_distr([
                0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
            ]),
        }


def kswin_distr_prelude() -> BaseDistribution[KSWIN]:
    return into_distr([
        {
            'type': KSWIN,
            **KSWIN.kwargs_distr_prelude(),
            'window_size': [stat_size * r for r in [2, 3, 4, 6]],
            'stat_size': stat_size,
        }
        for stat_size in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    ])


__all__ = [
    'ADWIN',
    'KSWIN',
    'PageHinkley',
    'kswin_distr_prelude',
]
