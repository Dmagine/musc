# This file is needed because of a pickle bug on Windows.
# See https://blog.csdn.net/m0_64007201/article/details/127677630.


import torch
import torch.nn as nn
import torch.nn.functional as F


def updator(model: nn.Module, x_arr: list[torch.Tensor], y_arr: list[torch.Tensor], dummy_lr: float) -> None:
    del dummy_lr
    model.load_state_dict(nn.Linear(2, 2).state_dict())
    x_arr_, y_arr_ = torch.stack(x_arr), torch.stack(y_arr)
    optim = torch.optim.Adam(model.parameters())
    # Iterate only 256 times to make HPO faster.
    for _ in range(256):
        y_pred_arr = model(x_arr_)
        loss = F.mse_loss(y_pred_arr, y_arr_)
        loss.backward()
        optim.step()
        optim.zero_grad()
