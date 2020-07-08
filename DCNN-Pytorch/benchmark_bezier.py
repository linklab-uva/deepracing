import torch
torch.backends.cudnn.enabled = False
from deepracing_models.nn_models.Models import AdmiralNetCurvePredictor as AdmiralNetCurvePredictor
import time
print("Building the network")
network = AdmiralNetCurvePredictor(params_per_dimension=7)
network = network.double().cuda(0)
running = True
print("Running the network in a loop")
while running:
  try:
      inp = torch.randn(1,5,3,66,200, device=torch.device("cuda:0"), dtype=torch.float64)
      output = network(inp)
      time.sleep(0.1)
  except KeyboardInterrupt as e:
      running = False


