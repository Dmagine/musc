import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from musc.high_level import (
    Metric,
    TorchDevicePool,
    UpdateStrategyByDriftDetection,
    UpdateStrategySearch,
)

from examples.hpo_updator import updator


def generate_mock_data(ground_truth_weight: Any, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_arr = torch.rand(n, 2)
    y_arr = x_arr.matmul(torch.tensor(ground_truth_weight).T)
    y_arr += torch.rand_like(y_arr) / 16.0
    return x_arr, y_arr


# Due to musc.hpo multiprocessing-based implementation, a main function here is mandatory.

def main() -> None:

    logging.getLogger().setLevel(logging.INFO)

    model = nn.Linear(2, 2)
    ground_truth_weight_old = [[-1.0, 2.0], [3.0, -4.0]]
    ground_truth_weight_new = [[0.0, 2.0], [3.0, -4.0]]

    x_arr, y_arr = generate_mock_data(ground_truth_weight_old, 256)
    updator(model, list(x_arr), list(y_arr), 0.0)

    x_arr_test_old, y_arr_test_old = generate_mock_data(ground_truth_weight_old, 4096)
    x_arr_test_new, y_arr_test_new = generate_mock_data(ground_truth_weight_new, 8192)
    x_arr = torch.cat([x_arr_test_old, x_arr_test_new])
    y_arr = torch.cat([y_arr_test_old, y_arr_test_new])
    t_x_arr = [0.0] * 4096 + [120.0] * 4096 + [240.0] * 4096
    t_y_arr = [60.0] * 4096 + [180.0] * 4096 + [300.0] * 4096

    search = UpdateStrategySearch(
        {
            'type': UpdateStrategyByDriftDetection,
            'updator': {
                'base_fn': updator,
                'dummy_lr': [0.0001, 0.001],
            },
            'data_amount_required': [128, 256],
            'metric': Metric(F.mse_loss, pred_first=True),
        },
        model,
        x_arr,
        y_arr,
        t_x_arr,
        t_y_arr,
        [Metric(F.mse_loss, pred_first=True), 'n_samples'],
        ['min', 'min'],
        top_k_kept=1000000,
        n_jobs=2,
        resource_pool=TorchDevicePool([0], 2),
        verbose_level=1,
        trace_history_file_path='example_hpo_trace_history.txt',
        optimal_scores_csv_path='example_hpo_optimal_scores.csv',
        optimal_samples_file_path='example_hpo_optimal_samples.txt',
        top_k_scores_csv_path='example_hpo_top_k_scores.csv',
        top_k_samples_file_path='example_hpo_top_k_samples.txt',
        load_old_state=True,
        stop_signal_file_path='example_hpo_stop_signal.txt',
    )

    with search:
        search.search(16)


if __name__ == '__main__':
    main()
