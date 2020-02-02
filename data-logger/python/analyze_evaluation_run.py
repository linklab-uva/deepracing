
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
parser.add_argument("dataset_dir", help="Directory of the dataset",  type=str)
args = parser.parse_args()
dset_dir = args.dataset_dir


failuredistances, failuretimes, failuretimediffs = deepracing.evaluation_utils.evalDataset(dset_dir,\
                                             "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=True)