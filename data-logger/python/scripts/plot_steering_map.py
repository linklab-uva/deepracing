import TimestampedUDPData_pb2
import google.protobuf.json_format
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import cv2
def sortkey(packet):
    return packet.udp_packet.m_time
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


recording_directree = 'C:\\Users\\ttw2x\\Documents\\source_builds\\deepracing\\data-logger\\build\\Release\\udp_data'
recording_packets = sorted(getAllPacketz(recording_directree), key=sortkey)
input_steering = []
computed_steering = []
L=3.65
for packet in recording_packets:
    omega = packet.udp_packet.m_ang_vel_y
    v = packet.udp_packet.m_speed
    input_proportion = packet.udp_packet.m_steer
    if(v<20 or np.abs(input_proportion) < 0.2):
        continue
    angle = np.arcsin(L*omega/v)
    input_steering.append(input_proportion)
    computed_steering.append(angle)

fig = plt.figure("Output Angle vs Proportion")
plt.plot(input_steering, computed_steering, label='steering line')
fig.legend()
plt.show()


# with open(filename, 'r') as file:
#     jsonstring = file.read()
# data = TimestampedUDPData_pb2.TimestampedUDPData()
# google.protobuf.json_format.Parse(jsonstring,data)
# print(data)