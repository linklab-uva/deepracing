
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
def udpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
def contiguous_regions(condition):
    """Finds contiguous True regions of the 1D boolean array "condition".
    Returns a 2D array where the first column is the start index of the region
    and the second column is the end index."""
    # Find the indicies of changes in "condition"
    idx = np.flatnonzero(np.diff(condition)) + 1

    # Prepend or append the start or end indicies to "idx"
    # if there's a block of "True"'s at the start or end...
    if condition[0]:
        idx = np.append(0, idx)
    if condition[-1]:
        idx = np.append(idx, len(condition))

    return idx.reshape(-1, 2)
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
mtbf= np.zeros(runmax)
mean_failure_distances = np.zeros(runmax)
num_failures = np.zeros(runmax)
for (i, dset) in enumerate(bezier_dsets):
    dset_dir = os.path.join(main_dir, dset)
    failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
        "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
    mtbf[i] = np.mean(failuretimediffs)
    mean_failure_distances[i] = np.mean(failuredistances)
    num_failures[i] = float(failuredistances.shape[0])
    # print( "Number of failures: %d" % ( num_failures[i] ) )
    # print( "Mean time between failures: %f" % ( mtbf[i] ) )
    # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
print("Results for Bezier Predictor:")
print( "Average Number of failures: %d" % ( np.mean(num_failures) ) )
print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) )
print( "Overall Mean failure distance: %f" % (  np.mean(mean_failure_distances)  ) )




for (i, dset) in enumerate(waypoint_dsets):
    dset_dir = os.path.join(main_dir, dset)
    failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
        "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
    mtbf[i] = np.mean(failuretimediffs)
    mean_failure_distances[i] = np.mean(failuredistances)
    num_failures[i] = float(failuredistances.shape[0])
    # print( "Number of failures: %d" % ( num_failures[i] ) )
    # print( "Mean time between failures: %f" % ( mtbf[i] ) )
    # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
print("Results for Waypoint Predictor:")
print( "Average Number of failures: %d" % ( np.mean(num_failures) ) )
print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) )
print( "Overall Mean failure distance: %f" % (  np.mean(mean_failure_distances)  ) )



for (i, dset) in enumerate(cnnlstm_dsets):
    dset_dir = os.path.join(main_dir, dset)
    failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
        "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
    mtbf[i] = np.mean(failuretimediffs)
    mean_failure_distances[i] = np.mean(failuredistances)
    num_failures[i] = float(failuredistances.shape[0])
    # print( "Number of failures: %d" % ( num_failures[i] ) )
    # print( "Mean time between failures: %f" % ( mtbf[i] ) )
    # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
print("Results for CNNLSTM:")
print( "Average Number of failures: %d" % ( np.mean(num_failures) ) )
print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) )
print( "Overall Mean failure distance: %f" % (  np.mean(mean_failure_distances)  ) )



for (i, dset) in enumerate(pilotnet_dsets):
    dset_dir = os.path.join(main_dir, dset)
    failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
        "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
    mtbf[i] = np.mean(failuretimediffs)
    mean_failure_distances[i] = np.mean(failuredistances)
    num_failures[i] = float(failuredistances.shape[0])
    # print( "Number of failures: %d" % ( num_failures[i] ) )
    # print( "Mean time between failures: %f" % ( mtbf[i] ) )
    # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
print("Results for PilotNet:")
print( "Average Number of failures: %d" % ( np.mean(num_failures) ) )
print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) )
print( "Overall Mean failure distance: %f" % (  np.mean(mean_failure_distances)  ) )