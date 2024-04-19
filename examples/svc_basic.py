import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from musc.high_level import Metric, Service, UpdateStrategyByDriftDetection
from river.drift import ADWIN

print('Entering preparation stage...')

model = nn.Linear(2, 2)
ground_truth_weight_old = [[-1.0, 2.0], [3.0, -4.0]]

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

svc = Service(
    model,
    UpdateStrategyByDriftDetection(
        ADWIN(),
        updator,
        256,
        Metric(F.mse_loss, pred_first=True),
    ),
)

print('Entering online stage 1...')

x_arr_1, y_arr_1 = generate_mock_data(ground_truth_weight_old, 4096)
y_pred_arr_1 = []

for i in range(4096):
    y_pred_arr_1.append(svc.recv_x(x_arr_1[i], i))
print(F.mse_loss(torch.stack(y_pred_arr_1), y_arr_1).tolist())

for i in range(4096):
    svc.recv_y(y_arr_1[i], i)
time.sleep(60.0)

print('Entering online stage 2...')

ground_truth_weight_new = [[0.0, 2.0], [3.0, -4.0]]

x_arr_2, y_arr_2 = generate_mock_data(ground_truth_weight_new, 4096)
y_pred_arr_2 = []

for i in range(4096):
    y_pred_arr_2.append(svc.recv_x(x_arr_2[i], 4096+i))
print(F.mse_loss(torch.stack(y_pred_arr_2), y_arr_2).tolist())

for i in range(4096):
    svc.recv_y(y_arr_2[i], 4096+i)
time.sleep(60.0)

print('Entering online stage 3...')

x_arr_3, y_arr_3 = generate_mock_data(ground_truth_weight_new, 4096)
y_pred_arr_3 = []

for i in range(4096):
    y_pred_arr_3.append(svc.recv_x(x_arr_3[i], 8192+i))
print(F.mse_loss(torch.stack(y_pred_arr_3), y_arr_3).tolist())

for i in range(4096):
    svc.recv_y(y_arr_3[i], 8192+i)
time.sleep(60.0)

print(f'Model update record(s): {svc.stats().model_update_records}')

# Outputs:
# Entering preparation stage...
# Updating model...
# Entering online stage 1...
# 0.0003353709471412003
# Entering online stage 2...
# 0.17006048560142517
# Updating model...
# Entering online stage 3...
# 0.00032756535802036524
# Model update record(s): [ModelUpdateRecord(drift_point=4192, new_concept_range=(4192, 4448), time_spent=23.510266304016113)]
