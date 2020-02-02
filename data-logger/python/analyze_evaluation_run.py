
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import argcomplete
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import deepracing.protobuf_utils as proto_utils
import scipy
import scipy.spatial
import deepracing.evaluation_utils
def analyzedatasets(main_dir,subdirs,prefix):
    mtbf= np.zeros(runmax)
    mean_failure_distances = np.zeros(runmax)
    num_failures = np.zeros(runmax)
    for (i, dset) in enumerate(subdirs):
        print("Running dataset %d for %s:"%(i+1, prefix))
        dset_dir = os.path.join(main_dir, dset)
        failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
            "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
        mtbf[i] = np.mean(failuretimediffs)
        mean_failure_distances[i] = np.mean(failuredistances)
        num_failures[i] = float(failuredistances.shape[0])
        # print( "Number of failures: %d" % ( num_failures[i] ) )
        # print( "Mean time between failures: %f" % ( mtbf[i] ) )
        # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
    print("Results for %s:"%(prefix))
    print( "Average Number of failures: %d" % ( np.mean(num_failures) ) )
    print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) )
    print( "Overall Mean failure distance: %f" % (  np.mean(mean_failure_distances)  ) )
parser = argparse.ArgumentParser()
parser.add_argument("main_dir", help="Directory of the evaluation datasets",  type=str)
args = parser.parse_args()
main_dir = args.main_dir

runmax = 2
plot=False
bezier_dsets = ["bezier_predictor_run%d" % i for i in range(1,runmax+1)]
waypoint_dsets = ["waypoint_predictor_run%d" % i for i in range(1,runmax+1)]
cnnlstm_dsets = ["cnnlstm_run%d" % i for i in range(1,runmax+1)]
pilotnet_dsets = ["pilotnet_run%d" % i for i in range(1,runmax+1)]
print(bezier_dsets)
print(waypoint_dsets)
print(cnnlstm_dsets)
print(pilotnet_dsets)




analyzedatasets(main_dir,bezier_dsets,"Bezier Predictor")
analyzedatasets(main_dir,waypoint_dsets,"Waypoint Predictor")
analyzedatasets(main_dir,cnnlstm_dsets,"CNNLSTM")
analyzedatasets(main_dir,pilotnet_dsets,"PilotNet")

