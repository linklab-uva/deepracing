import TimestampedUDPData_pb2
import google.protobuf.json_format
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import cv2
import scipy.stats as stats
import TimestampedPacketMotionData_pb2
def sortkey(packet : TimestampedPacketMotionData_pb2.TimestampedPacketMotionData):
    return packet.udp_packet.m_header.m_sessionTime
def getAllPacketz(directree : str):
    files = [os.path.join(directree,f) for f in os.listdir(directree) 
        if os.path.isfile(os.path.join(directree, f)) and str.lower(os.path.join(directree, f).split(".")[-1])=="json"]
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


recording_directree = 'C:\\Users\\ttw2x\\Documents\\git_repos\\deepracing\\data-logger\\build\\Release\\udp_data\\motion_packets'
recording_packets = sorted(getAllPacketz(recording_directree), key=sortkey)
delta = []
xvar = []
L=3.65
for packet in recording_packets:
    delta_ = np.array((packet.udp_packet.m_angularVelocityX, packet.udp_packet.m_angularVelocityY, packet.udp_packet.m_angularVelocityZ))
    delta.append( delta_[1] )
    xvar.append(  packet.udp_packet.m_localVelocityZ  * np.tan(-1.0 * packet.udp_packet.m_frontWheelsAngle  ) )

deltaarr = np.array(delta)
xvararr = np.array(xvar)
slope, intercept, r_value, p_value, std_err = stats.linregress(xvararr,deltaarr)
fitvals = slope * xvararr + intercept

print( "Computed Wheelbase: %f" % ( 1/slope ) )
print( "R2 value: %f" % ( r_value**2 ) )
print( "Standard Error: %f" % ( std_err ) )
fig = plt.figure("Output Angle vs Proportion")
plt.plot(xvar, delta, label='data')
plt.plot(xvar, fitvals, label='best fit line')
fig.legend()
plt.show()
