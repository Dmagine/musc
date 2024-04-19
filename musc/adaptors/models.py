from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from musc.service.concepts.a import BaseModel


__all__ = []


try:

    import torch

except ImportError:

    pass

else:

    class ModelPyTorch(BaseModel):

        def __init__(self, model: torch.nn.Module) -> None:
            if (dtype := ModelPyTorch._determine_dtype(model)) is None:
                raise ValueError
            if (device := ModelPyTorch._determine_device(model)) is None:
                raise ValueError
            self._model = model
            self._dtype = dtype
            self._device = device

        def __del__(self) -> None:
            try:
                torch.cuda.empty_cache()  # To avoid memory leak
            except AttributeError:
                # AttributeError: 'NoneType' object has no attribute 'empty_cache'
                pass

        @staticmethod
        def _determine_dtype(model: torch.nn.Module) -> torch.dtype | None:
            dtype = None
            for p in model.parameters():
                if dtype is None:
                    dtype = p.dtype
                elif p.dtype != dtype:
                    return None
            return dtype

        @staticmethod
        def _determine_device(model: torch.nn.Module) -> torch.device | None:
            device = None
            for p in model.parameters():
                if device is None:
                    device = p.device
                elif p.device != device:
                    return None
            return device

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                x = x.to(device=self._device)
                training_before = self._model.training
                self._model.eval()
                try:
                    return self._model(torch.stack([x]))[0]
                finally:
                    self._model.train(training_before)

        def inner(self) -> torch.nn.Module:
            return self._model

        def clone_with_resource(self, resource: Any) -> ModelPyTorch:
            if not isinstance(resource, torch.device):
                raise TypeError
            return ModelPyTorch(deepcopy(self._model).to(device=resource))

        def resource_current(self) -> Any:
            return self._device

        def resource_cpu(self) -> Any:
            return torch.device('cpu')

        def preprocess_x(self, x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.to(dtype=self._dtype)
            return torch.tensor(x, dtype=self._dtype)

        def preprocess_y(self, y: Any) -> torch.Tensor:
            if isinstance(y, torch.Tensor):
                return y.to(dtype=self._dtype)
            return torch.tensor(y, dtype=self._dtype)

        def y_pred_to_json_like(self, y_pred: torch.Tensor) -> Any:
            return y_pred.tolist()

    __all__.append('ModelPyTorch')


try:

    import sklearn as _

except ImportError:

    pass

else:

    class ModelScikitLearn(BaseModel):

        def __init__(self, model: Any) -> None:
            self._model = model

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self._model.predict(np.array([x]))[0]

        def inner(self) -> Any:
            return self._model

        def preprocess_x(self, x: Any) -> np.ndarray:
            # See https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
            # "The default data type in NumPy is float_."
            return np.array(x, dtype=np.float_)

        def preprocess_y(self, y: Any) -> np.ndarray:
            # See https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
            # "The default data type in NumPy is float_."
            return np.array(y, dtype=np.float_)

        def y_pred_to_json_like(self, y_pred: np.ndarray) -> Any:
            return y_pred.tolist()

    __all__.append('ModelScikitLearn')
