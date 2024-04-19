import torch.nn as nn
from musc.high_level import UpdateStrategyNone, run_service_http

model = nn.Linear(2, 2)
run_service_http(model, UpdateStrategyNone())

# $ curl -X POST localhost -H 'Content-Type: application/json' -d '{ "id": 0, "x": [0, 1] }'
# {"y_pred":[-0.9937736988067627,0.016456782817840576]}
