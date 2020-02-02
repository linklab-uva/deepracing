
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
def evalDataset(dset_dir, inner_trackfile, outer_trackfile, plot = False):
    image_dir = os.path.join(dset_dir,"images")
    udp_dir = os.path.join(dset_dir,"udp_data")
    motion_dir = os.path.join(udp_dir,"motion_packets")
    motion_packets = sorted(proto_utils.getAllMotionPackets(motion_dir, False), key = udpPacketKey)
    positions = np.array([proto_utils.extractPosition(p.udp_packet) for p in motion_packets])
    xypoints = positions[:,[0,2]]

    rinner, Xinner = proto_utils.loadTrackfile(inner_trackfile)
    router, Xouter = proto_utils.loadTrackfile(outer_trackfile)
    
    outerKdTree = scipy.spatial.KDTree(Xouter[:,[0,2]])
    innerKdTree = scipy.spatial.KDTree(Xinner[:,[0,2]])

    innerpolygon : Polygon = Polygon(Xinner[:,[0,2]].tolist())
    outerpolygon : Polygon = Polygon(Xouter[:,[0,2]].tolist())
    if plot:
        plt.plot(xypoints[:,0], xypoints[:,1])
        plt.plot(*innerpolygon.exterior.xy)
        plt.plot(*outerpolygon.exterior.xy)
        plt.show()

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
    sessiontime_array = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets])
    sessiontime_array = sessiontime_array - sessiontime_array[0]
    distancearray = np.array(distancelist)
    failureregions = np.array(contiguous_regions(offtrackarr))
    numfailures = failureregions.shape[0]

    if numfailures>0:
        # print("Went off track %d times" %(numfailures,) )
        failuredistances = np.array([np.mean(distancearray[failureregions[i,0]:failureregions[i,1]])  for i in range(failureregions.shape[0])])
        failuretimes = np.array([(sessiontime_array[failureregions[i,0]],sessiontime_array[failureregions[i,1]-1] )  for i in range(failureregions.shape[0])])
        failuretimediffs = np.array([failuretimes[0,0]]+[(failuretimes[i+1,0]-failuretimes[i,1]) for i in range(0,failuretimes.shape[0]-1)])
        # print(failuretimes)
        # print(failuretimediffs)
        if plot:
            for i in range(failureregions.shape[0]):
                Pathfail = xypoints[failureregions[i,0]:failureregions[i,1]]
                distances_inner, innerfailIndices = innerKdTree.query(Pathfail)
                distances_outer, outerfailIndices = outerKdTree.query(Pathfail)
                Xouterfail = Xouter[outerfailIndices]
                Xinnerfail = Xinner[innerfailIndices]
                plt.plot(Xouterfail[:,0], Xouterfail[:,2], label="Outer Boundary", color="r")
                plt.plot(Xinnerfail[:,0], Xinnerfail[:,2], label="Inner Boundary", color="g")
                plt.plot(Pathfail[:,0], Pathfail[:,1], label="Followed Path", color="b")
                plt.legend()
                plt.show()
        return failuredistances, failuretimes, failuretimediffs
    else:
        return None, None, None