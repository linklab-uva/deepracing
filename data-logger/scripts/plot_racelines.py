import TimestampedUDPData_pb2
import google.protobuf.json_format
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
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

        
playback_directree = 'D:\\test_data\\grand_prix_usa\\playback'
recording_directree = 'D:\\test_data\\grand_prix_usa\\udp_data'
playback_packets = sorted(getAllPacketz(playback_directree), key=sortkey)
#print(playback_packets[-1])
recording_packets = sorted(getAllPacketz(recording_directree), key=sortkey)
#print(recording_packets[-1])



for i in range(305):
    print(recording_packets[i].udp_packet.m_lapTime)
first_recording_zero = findFirstZero(recording_packets)
print(recording_packets[first_recording_zero])
playback_raceline, playback_times = extractDataz(playback_packets, first_zero=0)


first_recording_zero = findFirstZero( recording_packets )
recording_raceline, recording_times = extractDataz( recording_packets, first_zero=first_recording_zero )
cutofftime = np.max( recording_times )
I = playback_times<cutofftime
playback_raceline=playback_raceline[ I ]
playback_times=playback_times[ I ]
print("Got %d points out of %d samples from playback" % (playback_times.shape[0],len(playback_packets)))
print("Got %d points out of %d samples from recording" % (recording_times.shape[0],len(recording_packets)))

x_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,0] )
y_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,1] )
z_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,2] )
steer_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,3] )
throttle_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,4] )
brake_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,5] )
recording_interpolants = np.stack( (x_interpolants,y_interpolants,z_interpolants, steer_interpolants, throttle_interpolants, brake_interpolants), axis=1 )

playback_steering = playback_raceline[:,3]
recording_steering = recording_raceline[:,3]
fig = plt.figure("Steering")
plt.plot(playback_times, playback_steering, label='Playback')
plt.plot(recording_times, recording_steering, label='Recording')
fig.legend()

playback_throttle = playback_raceline[:,4]
recording_throttle = recording_raceline[:,4]
fig = plt.figure("Throttle")
plt.plot(playback_times, playback_throttle, label='Playback')
plt.plot(recording_times, recording_throttle, label='Recording')
fig.legend()



playback_brake = playback_raceline[:,5]
recording_brake = recording_raceline[:,5]
fig = plt.figure("Brake")
plt.plot(playback_times, playback_brake, label='Playback')
plt.plot(recording_times, recording_brake, label='Recording')
fig.legend()




diffs = playback_steering - steer_interpolants
absdiffs = np.abs(diffs)
fig = plt.figure("Diffs")
plt.plot(playback_times, diffs)
plt.show()




# with open(filename, 'r') as file:
#     jsonstring = file.read()
# data = TimestampedUDPData_pb2.TimestampedUDPData()
# google.protobuf.json_format.Parse(jsonstring,data)
# print(data)