import TimestampedUDPData_pb2
import TimestampedPacketMotionData_pb2
import google.protobuf.json_format
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import cv2
from scipy.spatial import KDTree as KDTree
def sortkey(packet):
    return packet.udp_packet.m_header.m_sessionTime
def getAllPacketz(directree : str, subdir : str = "motion_packets"):
    files = [os.path.join(directree,subdir,f) for f in os.listdir(os.path.join(directree,subdir)) 
        if os.path.isfile(os.path.join(directree,subdir, f)) and str.lower(os.path.join(directree, subdir, f).split(".")[-1])=="json"]
    rtn = []
    for filename in files:
        with open(filename, 'r') as file:
            jsonstring = file.read()
        data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
        google.protobuf.json_format.Parse(jsonstring,data)
        rtn.append(data)
    return rtn
def extractDataz(packets: list, first_zero : int =0):
    arr = np.zeros((len(packets)-first_zero,6))
    times = np.zeros(len(packets)-first_zero)
    for idx in range(first_zero,len(packets)):
        arr[idx-first_zero][0]=packets[idx].udp_packet.m_x
        arr[idx-first_zero][1]=packets[idx].udp_packet.m_y
        arr[idx-first_zero][2]=packets[idx].udp_packet.m_z
        arr[idx-first_zero][3]=packets[idx].udp_packet.m_steer
        arr[idx-first_zero][4]=packets[idx].udp_packet.m_throttle
        arr[idx-first_zero][5]=packets[idx].udp_packet.m_brake
        times[idx-first_zero]=packets[idx].udp_packet.m_lapTime
    return arr, times
def findFirstZero(packets: list):
    for idx in range(len(packets)):
        if packets[idx].udp_packet.m_lapTime<1E-2:
            return idx
        # else:
        #     print(packets[idx].udp_packet.m_lapTime)
    raise AttributeError("List of packets has no laptime of zero.")

image_directree = 'D:\\test_data\\australia_purepursuit\\images'
recording_directree = 'D:\\test_data\\australia_purepursuit\\udp_data'
raceline_arma_file = 'D:\\test_data\\australia_purepursuit\\Australia_racingline.arma.txt'
image = cv2.imread(os.path.join(image_directree,'image_500.jpg'))
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
recording_packets = sorted(getAllPacketz(recording_directree), key=sortkey)
motion_data = [  packet.udp_packet.m_carMotionData[0] for packet in recording_packets]
path = np.array([  [motion_data_.m_worldPositionX, motion_data_.m_worldPositionY, motion_data_.m_worldPositionZ] for motion_data_ in motion_data])
print(path.shape)
raceline = np.loadtxt(raceline_arma_file,dtype=np.float64,delimiter="\t",skiprows=2)
raceline_path = raceline[:,1:4]
raceline_s = raceline_path[:,0]
print(raceline.shape)
print(raceline_path.shape)
colors = ['r', 'g']
markers = ['o', 'o']
kdtree = KDTree(raceline_path, leafsize=path.shape[0]+1)
distances, indices = kdtree.query(path, k=1)
nearest_neighbors = raceline_path[indices]
print(nearest_neighbors.shape)
print("Average distance from pure pursuit line to raceline: %f" %(np.mean(distances)))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(raceline_path[:,0], raceline_path[:,1], raceline_path[:,2], c=colors[0], marker=markers[0], s = np.ones_like(raceline_s))
ax.scatter(path[:,0], path[:,1], path[:,2], c=colors[1], marker=markers[1], s = np.ones_like(raceline_s))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[0],marker=markers[0])
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors[1], marker=markers[1])
ax.legend([scatter1_proxy, scatter2_proxy], ["Ideal Raceline", "Pure Pursuit path"], numpoints = 1)
fig2 = plt.figure()
distances_plot = plt.plot(distances, label='KD Tree Query Distances')
plt.legend()
plt.show()




#print(recording_packets[-1])




# with open(filename, 'r') as file:
#     jsonstring = file.read()
# data = TimestampedUDPData_pb2.TimestampedUDPData()
# google.protobuf.json_format.Parse(jsonstring,data)
# print(data)