
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import argcomplete
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing
import matplotlib.pyplot as plt
import deepracing.protobuf_utils as proto_utils
import deepracing.pose_utils as pose_utils
import scipy
import scipy.spatial
import numpy.linalg as la
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
def polyDist(polygon : Polygon, point : Point):
    if point.within(polygon):
        return polygon.exterior.distance(point)
    return point.distance(polygon)
    
def getRacePlots(dset_dir, json=False):
    motion_dir = os.path.join(dset_dir ,"udp_data","motion_packets")
    lap_dir = os.path.join(dset_dir ,"udp_data","lap_packets")
    telemetry_dir = os.path.join(dset_dir,"udp_data","car_telemetry_packets")
    motion_packets = sorted(proto_utils.getAllMotionPackets(motion_dir, json), key = udpPacketKey)
    lap_packets = sorted(proto_utils.getAllLapDataPackets(lap_dir, json), key = udpPacketKey)
    telemetry_packets = sorted(proto_utils.getAllTelemetryPackets(telemetry_dir, json), key = udpPacketKey)

    laptimes =  np.array([p.udp_packet.m_lapData[0].m_currentLapTime for p in lap_packets])
    laptimedt = np.diff(laptimes)
    drops = laptimedt<0
    if np.sum(drops.astype(np.uint32))==0:
        lapstart = 0
        lapend = len(laptimes)
    else:
        lapstart = np.argmax(drops)+1
        if np.sum(drops.astype(np.int32))>1:
            lapend = lapstart + np.argmax(drops[lapstart+1:]) + 2
        else:
            lapend = len(laptimes)
    print("Got %d laptimes" %(len(laptimes),))
    lapstart+=2
    lapend-=2
    print("lapstart index :%d" %( lapstart ,))
    print("lapend index :%d" %( lapend ,))
    print("Cropping to %d laptimes" %(lapend - lapstart,))
    sessiontime_lapstart = lap_packets[lapstart].udp_packet.m_header.m_sessionTime
    sessiontime_lapend = lap_packets[lapend].udp_packet.m_header.m_sessionTime
    laptimes = lap_packets[lapstart:lapend]
    poses = [proto_utils.extractPose(p.udp_packet) for p in motion_packets]
    poses_true = [proto_utils.extractPose(p.udp_packet)\
        for p in motion_packets if (p.udp_packet.m_header.m_sessionTime>=sessiontime_lapstart and p.udp_packet.m_header.m_sessionTime<=sessiontime_lapend)]
    positions = np.array([pose[0] for pose in poses])
    rotations = np.array([pose[1] for pose in poses])
    positions_true = np.array([pose[0] for pose in poses_true])
    rotations_true = np.array([pose[1] for pose in poses_true])
    distances = np.array([p.udp_packet.m_lapData[0].m_lapDistance\
         for p in lap_packets if (p.udp_packet.m_header.m_sessionTime>=sessiontime_lapstart and p.udp_packet.m_header.m_sessionTime<=sessiontime_lapend)])
    throttles = np.array([p.udp_packet.m_carTelemetryData[0].m_throttle\
         for p in telemetry_packets if (p.udp_packet.m_header.m_sessionTime>=sessiontime_lapstart and p.udp_packet.m_header.m_sessionTime<=sessiontime_lapend)])/100.0
    steering = -np.array([p.udp_packet.m_carTelemetryData[0].m_steer\
         for p in telemetry_packets if (p.udp_packet.m_header.m_sessionTime>=sessiontime_lapstart and p.udp_packet.m_header.m_sessionTime<=sessiontime_lapend)])/100.0
    velocities = 3.6*np.array([proto_utils.extractVelocity(p.udp_packet)\
         for p in motion_packets if (p.udp_packet.m_header.m_sessionTime>=sessiontime_lapstart and p.udp_packet.m_header.m_sessionTime<=sessiontime_lapend)])
    return distances, throttles, steering, velocities, positions, rotations, positions_true, rotations_true
def evalDataset(dset_dir, inner_trackfile, outer_trackfile, plot = False, json=False):
    image_dir = os.path.join(dset_dir,"images")
    udp_dir = os.path.join(dset_dir,"udp_data")
    motion_dir = os.path.join(udp_dir,"motion_packets")
    motion_packets = sorted(proto_utils.getAllMotionPackets(motion_dir, json), key = udpPacketKey)
    positions = np.array([proto_utils.extractPosition(p.udp_packet) for p in motion_packets])
    velocities = np.array([proto_utils.extractVelocity(p.udp_packet) for p in motion_packets])
    xypoints = positions[:,[0,2]]
    xydiffs = xypoints[1:] - xypoints[:-1]
    xydiffs = np.vstack((np.zeros(2), xydiffs))
    diffnorms = la.norm(xydiffs,axis=1)
    cummulativenormsums = np.cumsum(diffnorms)

    rinner, Xinner = proto_utils.loadTrackfile(inner_trackfile)
    router, Xouter = proto_utils.loadTrackfile(outer_trackfile)
    
    outerKdTree = scipy.spatial.KDTree(Xouter[:,[0,2]])
    innerKdTree = scipy.spatial.KDTree(Xinner[:,[0,2]])

    innerpolygon : Polygon = Polygon(Xinner[:,[0,2]].tolist())
    innerpolygon_ext = LinearRing(innerpolygon.exterior.coords)
    outerpolygon : Polygon = Polygon(Xouter[:,[0,2]].tolist())
    if plot:
        plt.plot(xypoints[:,0], xypoints[:,1])
        plt.plot(*innerpolygon.exterior.xy)
        plt.plot(*outerpolygon.exterior.xy)
        plt.show()

    offtrackarr = []
    distancelist = []
    for i in range(len(motion_packets)):
        point : Point = Point(xypoints[i].tolist())
        outside = not point.within(outerpolygon)
        inside = point.within(innerpolygon)
        offtrackarr.append(outside or inside)
        if inside:
            distancelist.append(innerpolygon.exterior.distance(point))
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

        failurescores = np.array([np.mean(distancearray[failureregions[i,0]:failureregions[i,1]])  for i in range(failureregions.shape[0])])

        
        failuretimes = np.array([(sessiontime_array[failureregions[i,0]],sessiontime_array[failureregions[i,1]-1] )  for i in range(failureregions.shape[0])])
        failuretimediffs = np.array([failuretimes[0,0]]+[(failuretimes[i+1,0]-failuretimes[i,1]) for i in range(0,failuretimes.shape[0]-1)])
        
        failuredistances = np.array([(cummulativenormsums[failureregions[i,0]],cummulativenormsums[failureregions[i,1]-1] )  for i in range(failureregions.shape[0])])
        failuredistancediffs = np.array([failuredistances[0,0]]+[(failuredistances[i+1,0]-failuredistances[i,1]) for i in range(0,failuredistances.shape[0]-1)])

        failurescores = failurescores[failurescores<1E10]
        # failuretimes = failuretimes[failurescores<1E10]
        # failuretimediffs = failuretimediffs[failurescores<1E10]
        # failuredistances = failuredistances[failurescores<1E10]
        # failuredistancediffs = failuredistancediffs[failurescores<1E10]
        
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
        return motion_packets, failurescores, failuretimes, failuretimediffs, failuredistances, failuredistancediffs, velocities, cummulativenormsums
    else:
        return motion_packets, None, None, None, None, None, velocities, cummulativenormsums