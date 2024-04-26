import musc.high_level as musc
import torch

model = torch.nn.Linear(2, 2)
musc.run_service_http(model, musc.UpdateStrategyBlank())

# $ curl -X POST localhost -H 'Content-Type: application/json' -d '{ "x": [0, 1], "id": 0 }'
# {"y_pred":[-0.9937736988067627,0.016456782817840576]}
