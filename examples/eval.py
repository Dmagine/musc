import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from musc.high_level import Evaluator, Metric, UpdateStrategyByDriftDetection
from river.drift import ADWIN

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

model = nn.Linear(2, 2)
ground_truth_weight_old = [[-1.0, 2.0], [3.0, -4.0]]
ground_truth_weight_new = [[0.0, 2.0], [3.0, -4.0]]

def generate_mock_data(ground_truth_weight: Any, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_arr = torch.rand(n, 2)
    y_arr = x_arr.matmul(torch.tensor(ground_truth_weight).T)
    y_arr += torch.rand_like(y_arr) / 16.0
    return x_arr, y_arr

def updator(model: nn.Module, x_arr: list[torch.Tensor], y_arr: list[torch.Tensor]) -> None:
    print('Updating model...')
    model.load_state_dict(nn.Linear(2, 2).state_dict())
    x_arr_, y_arr_ = torch.stack(x_arr), torch.stack(y_arr)
    optim = torch.optim.Adam(model.parameters())
    for _ in range(65536):
        y_pred_arr = model(x_arr_)
        loss = F.mse_loss(y_pred_arr, y_arr_)
        loss.backward()
        optim.step()
        optim.zero_grad()

x_arr, y_arr = generate_mock_data(ground_truth_weight_old, 256)
updator(model, list(x_arr), list(y_arr))

evaluator = Evaluator(
    model,
    UpdateStrategyByDriftDetection(
        ADWIN(),
        updator,
        256,
        Metric(F.mse_loss, pred_first=True),
    ),
)

x_arr_test_old, y_arr_test_old = generate_mock_data(ground_truth_weight_old, 4096)
x_arr_test_new, y_arr_test_new = generate_mock_data(ground_truth_weight_new, 8192)
x_arr = torch.cat([x_arr_test_old, x_arr_test_new])
y_arr = torch.cat([y_arr_test_old, y_arr_test_new])
t_x_arr = np.concatenate([
    np.arange(4096),
    np.arange(4096) + 65536 * 2,
    np.arange(4096) + 65536 * 4,
])
t_y_arr = np.concatenate([
    np.arange(4096) + 65536 * 1,
    np.arange(4096) + 65536 * 3,
    np.arange(4096) + 65536 * 5,
])

evaluation_start_time = time.time()
mse, stats = evaluator.evaluate(
    x_arr,
    y_arr,
    t_x_arr,
    t_y_arr,
    Metric(F.mse_loss, pred_first=True),
)
print(f'Evaluation time: {time.time() - evaluation_start_time} seconds')
print(f'MSE: {mse}')
print(f'Model update record(s): {stats.model_update_records}')

# Outputs:
# Updating model...
# Updating model...
# Evaluation time: 30.5738205909729 seconds
# MSE: 0.056748427225917504
# Model update record(s): [ModelUpdateRecord(drift_point=4192, new_concept_range=(4192, 4448), time_spent=29.01304268836975)]
