import TimestampedUDPData_pb2
import google.protobuf.json_format
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sortkey(packet):
    return packet.udp_packet.m_lapTime
def getAllPacketz(directree : str):
    files = [os.path.join(directree,f) for f in os.listdir(directree) 
        if os.path.isfile(os.path.join(directree, f)) and str.lower(os.path.join(directree, f).split(".")[-1])=="json"]
    rtn = []
    for filename in files:
        with open(filename, 'r') as file:
            jsonstring = file.read()
        data = TimestampedUDPData_pb2.TimestampedUDPData()
        google.protobuf.json_format.Parse(jsonstring,data)
        rtn.append(data)
    return rtn
def extractTimes(packets: list):
    arr = np.zeros(len(packets))
    for idx in range(len(packets)):
        arr[idx]=packets[idx].udp_packet.m_lapTime
    return arr
def extractPositions(packets: list):
    arr = np.zeros((len(packets),3))
    for idx in range(len(packets)):
        arr[idx][0]=packets[idx].udp_packet.m_x
        arr[idx][1]=packets[idx].udp_packet.m_y
        arr[idx][2]=packets[idx].udp_packet.m_z
    return arr
        
playback_directree = '/home/ttw2xk/deepf1data/usa_gp_playback_recalibrated2_1/playback'
recording_directree = '/home/ttw2xk/deepf1data/usa_gp_recalibrated2/udp_data'
playback_packets = sorted(getAllPacketz(playback_directree), key=sortkey)
print(playback_packets[-1])
recording_packets = sorted(getAllPacketz(recording_directree), key=sortkey)
print(recording_packets[-1])

playback_raceline = extractPositions(playback_packets)
playback_times = extractTimes(playback_packets)

recording_raceline = extractPositions(recording_packets)
recording_times = extractTimes(recording_packets)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(recording_raceline[:,0], recording_raceline[:,1], recording_raceline[:,2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#plt.plot(recording_times, recording_raceline, 'b')
#plt.axis([0, 6, 0, 20])
plt.show()


# with open(filename, 'r') as file:
#     jsonstring = file.read()
# data = TimestampedUDPData_pb2.TimestampedUDPData()
# google.protobuf.json_format.Parse(jsonstring,data)
# print(data)