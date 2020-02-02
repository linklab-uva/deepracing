
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import argcomplete
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import deepracing.protobuf_utils as proto_utils
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
image_dir = os.path.join(dset_dir,"images")
udp_dir = os.path.join(dset_dir,"udp_data")
motion_dir = os.path.join(udp_dir,"motion_packets")
motion_packets = sorted(proto_utils.getAllMotionPackets(motion_dir, False), key = udpPacketKey)
positions = np.array([proto_utils.extractPosition(p.udp_packet) for p in motion_packets])
xypoints = positions[:,[0,2]]

rinner, Xinner = proto_utils.loadTrackfile("../tracks/Australia_innerlimit.track")
router, Xouter = proto_utils.loadTrackfile("../tracks/Australia_outerlimit.track")
print(Xinner.shape)
print(Xouter.shape)

innerpolygon : Polygon = Polygon(Xinner[:,[0,2]].tolist())
outerpolygon : Polygon = Polygon(Xouter[:,[0,2]].tolist())

offtrackarr = []
distancelist = []
for i in range(len(motion_packets)):
    point : Point = Point(xypoints[i])
    outside = not point.within(outerpolygon)
    inside = point.within(innerpolygon)
    offtrackarr.append(outside or inside)
    if inside:
        distancelist.append(point.distance(innerpolygon))
    elif outside:
        distancelist.append(point.distance(outerpolygon))
    else:
        distancelist.append(0.0)
print(offtrackarr)
sessiontime_array = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets])
distancearray = np.array(distancelist)
failureregions = np.array(contiguous_regions(offtrackarr))
numfailures = failureregions.shape[0]
print("Went off track %d times" %(numfailures,) )
failuredistances = np.array([np.mean(distancearray[failureregions[i,0]:failureregions[i,1]]) \
                             for i in range(failureregions.shape[0])])
failuretimes = np.array([(sessiontime_array[failureregions[i,0]],sessiontime_array[failureregions[i,1]] ) \
                             for i in range(failureregions.shape[0])])




plt.plot(xypoints[:,0], xypoints[:,1])
plt.plot(*innerpolygon.exterior.xy)
plt.plot(*outerpolygon.exterior.xy)
plt.show()
