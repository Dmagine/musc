# musc

[![](https://img.shields.io/pypi/v/musc)](https://pypi.org/project/musc/)

musc (**m**odel **u**pdate **s**trategy **c**onstruction) is a Python library aiming to help users to construct concept drift adaptive model services.

Note: SemVer is not yet followed in 0.1.x versions.

## Preliminary

Q: What is concept drift?

A: Concept drift is the phenomenon that data distribution received by the deployed model changes over time. Machine learning algorithms usually assume that data received online has the same distribution as in the training set, so concept drift will make model performance become worse.

Q: What is model update strategy?

A: In this library, model update strategy determines **when** and **how** the model should be updated to adapt to the new data distribution, and is an essential part of our concept drift adaptive model services. For the "when" problem, the model update can be periodically, or based on concept drift detection algorithms. For the "how" problem, the model update can be done by retraining or fine-tuning using the latest labelled data received online.

## Usage

musc can be installed by `pip install musc`. musc requires Python 3.11 and above.

### Constructing model service

The simplest way to construct concept drift adaptive model service is to use the `musc.run_service_http` function:

```python
import musc.high_level as musc
import torch

model = torch.nn.Linear(2, 2)
musc.run_service_http(model, musc.UpdateStrategyBlank())
```

A model object and a model update strategy object is required. For model, currently supported model types include PyTorch's `torch.nn.Module` and scikit-learn's `sklearn.base.BaseEstimator`. If your model does not belong to any of these, you can implement your `musc.BaseModel` subclass and use it. For model update strategy, an `musc.UpdateStrategy(Blank|ByPeriodicallyUpdate|ByDriftDetection)` object is expected. A "blank" strategy is used here for simplicity, which means that the model will never be updated, and more details about mode update strategy will be covered later.

The constructed model service can be called by cURL:

```
$ curl -X POST localhost -H 'Content-Type: application/json' -d '{ "x": [0, 1], "id": 0 }'
{"y_pred":[-0.9937736988067627,0.016456782817840576]}
$ curl -X POST localhost -H 'Content-Type: application/json' -d '{ "y": [0, 0], "id": 0 }'
{}
```

If called with an input value, then the prediction will be returned. If called with a ground truth label, then the label will be passed to the model update strategy. The "id" argument is required for the model service to identify the corresponding input value when receiving a label.

If you wish to embed the constructed model service into other HTTP/GRPC/... services instead of making it an independent one, you can use the `musc.Service` class instead of the `musc.run_service_http` function:

```python
svc = musc.Service(model, musc.UpdateStrategyBlank())

y_pred = svc.recv_x(torch.tensor([0.0, 1.0]), id_=0)  # tensor([0.4906, 0.5624])
_      = svc.recv_y(torch.tensor([0.0, 0.0]), id_=0)
```

### Constructing model update strategy

In addition to the "blank" strategy, an easy-to-understand kind of model update strategy is the periodically update strategy:

```python
def updator(model, x_arr, y_arr, lr=1e-4):
    ...  # In-place retraining or fine-tuning

strategy = musc.UpdateStrategyByPeriodicallyUpdate(period=32, updator=updator)
```

To construct this kind of strategy, you need to specify update period by number of samples and a "model updator" object, which accepts three arguments: the model object that should be updated in-place, and the x and y array of the latest labelled data received by the model service, whose length equal to the update period. See [examples/svc_basic.py](examples/svc_basic.py) for an example of model updator.

A more complex kind of model update strategy is the drift detection update strategy:

```python
from river.drift import ADWIN  # A drift detection algorithm

metric = musc.Metric(torch.nn.functional.mse_loss, pred_first=True)

strategy = musc.UpdateStrategyByDriftDetection(
    drift_detector=ADWIN(),
    metric=metric,
    updator=updator,
    data_amount_required=32,
)
```

To construct this kind of strategy, instead of specifying the update period, you need to specify a concept drift detector from [River](https://github.com/online-ml/river) and a `musc.Metric` object which calculates metric value for each pair of prediction and label, which is needed by the drift detector. You also need to tell the model service how much data is needed within a model update by the `data_amount_required` argument, so that after the drift detector reports a drift, the model service can collect an appropriate amount of labelled data for your model updator.

### Evaluation of model update strategy

To make a comparison between model update strategies before model service construction, you can perform evaluation for your model update strategy candidates:

```python
import pandas as pd

test_data = pd.read_csv(...)
x_arr, y_arr, t_x_arr, t_y_arr = test_data[...].to_numpy(), ...

eval_ = musc.Evaluator(model, strategy)
mse, stats = eval_.evaluate(x_arr, y_arr, t_x_arr, t_y_arr, metric)

print(mse)
print(stats.model_update_cost_by_time())
print(stats.model_update_cost_by_num_updates())
print(stats.model_update_cost_by_num_samples())
```

To perform evaluation, you need to provide test data that can represent what the model service will receive. Test data should contain input values and ground truth labels, along with their arrival timestamps. Evaluation will give you information about model performance and model update cost, allowing you to make a good trade-off between the two. See [examples/eval.py](examples/eval.py) for more details about model update strategy evaluation.

### Hyperparameter optimization of model update strategy

If you are not familiar with the field of concept drift and have no idea about how to select the potentially ideal model update strategy type (for example, by periodically update or by drift detection, by retraining or by fine-tuning) and parameter values (such as period of periodically update and learning rate of fine-tuning), you can perform hyperparameter optimization to find model update strategies that work well in the target scenario, without the need to understand how drift detectors work:

```python
search_space = {
    'type': musc.UpdateStrategyByDriftDetection,
    'metric': metric,
    'updator': {
        'base_fn': updator,
        'lr': [1e-4, 1e-3, 1e-2, 1e-1],
    },
    'data_amount_required': [16, 32, 64, 128],
}

if __name__ == '__main__':
    search = musc.UpdateStrategySearch(
        search_space, model, x_arr, y_arr, t_x_arr, t_y_arr,
        metric=[metric, 'time'], optim_mode=['min', 'min'],
        top_k_scores_csv_path='top_k_scores.csv',
        top_k_samples_file_path='top_k_samples.txt',
    )
    search.search(10000)
```

Note that in the above example the search space of drift detector is not specified. In case of this a reasonable "default search space" will be used, so that the ideal drift detector can be searched without user knowledge. During the search, the evaluated model update strategy samples and their metric values will be outputted to files, sorted by the first metric specified. See [examples/hpo.py](examples/hpo.py) for more details about model update strategy hyperparameter optimization.

## Highlights

* The constructed model service is robust against abnormal cases like ground truth absence.

* The evaluation is more accurate than existing approaches.

* The hyperparameter optimization support utilizing multiple GPUs.

## Acknowledgement

This project is supported by the National Key R&D Program of China (2021YFB1715200).
