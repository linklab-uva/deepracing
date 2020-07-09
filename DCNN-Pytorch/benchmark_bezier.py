import torch
import argparse
parser = argparse.ArgumentParser(description="Benchmark the bezier predictor on the gpu")
parser.add_argument("--cudnn", action="store_true")
args = parser.parse_args() 
argdict = vars(args)
torch.backends.cudnn.enabled = argdict["cudnn"]
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
      tick = time.time()
      output = network(inp)
      tock = time.time()
      print( "Ran the network in %f seconds" % ( tock - tick, ) )
      time.sleep(0.1)
  except KeyboardInterrupt as e:
      running = False


