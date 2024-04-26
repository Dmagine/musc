from musc.eval.high_level import Evaluator
from musc.hpo.distr.high_level import SingleValue, into_distr
from musc.hpo.high_level.a import UpdateStrategySearch
from musc.service.concepts.a import BaseModel, Metric
from musc.service.concepts.high_level.a import (
    UpdateStrategyByDriftDetection,
    UpdateStrategyByPeriodicallyUpdate,
    UpdateStrategyBlank,
)
from musc.service.high_level import Service, run_service_http

from . import drift_detectors

__all__ = [
    'BaseModel',
    'Evaluator',
    'Metric',
    'Service',
    'SingleValue',
    'UpdateStrategyByDriftDetection',
    'UpdateStrategyByPeriodicallyUpdate',
    'UpdateStrategyBlank',
    'UpdateStrategySearch',
    'drift_detectors',
    'into_distr',
    'run_service_http',
]

try:
    import torch as _
except ImportError:
    pass
else:
    from musc.adaptors.models import ModelPyTorch
    from musc.hpo.res_pool.torch_device import TorchDevicePool
    __all__.extend(['ModelPyTorch', 'TorchDevicePool'])

try:
    import sklearn as _
except ImportError:
    pass
else:
    from musc.adaptors.models import ModelScikitLearn
    __all__.extend(['ModelScikitLearn'])
