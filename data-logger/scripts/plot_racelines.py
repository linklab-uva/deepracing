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
def extractPositions(packets: list, first_zero : int =0):
    arr = np.zeros((len(packets)-first_zero,3))
    times = np.zeros(len(packets)-first_zero)
    for idx in range(first_zero,len(packets)):
        arr[idx-first_zero][0]=packets[idx].udp_packet.m_x
        arr[idx-first_zero][1]=packets[idx].udp_packet.m_y
        arr[idx-first_zero][2]=packets[idx].udp_packet.m_z
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
playback_raceline, playback_times = extractPositions(playback_packets, first_zero=0)


first_recording_zero = findFirstZero( recording_packets )
recording_raceline, recording_times = extractPositions( recording_packets, first_zero=first_recording_zero )
cutofftime = np.max( recording_times )
I = playback_times<cutofftime
playback_raceline=playback_raceline[ I ]
playback_times=playback_times[ I ]
print("Got %d points out of %d samples from playback" % (playback_times.shape[0],len(playback_packets)))
print("Got %d points out of %d samples from recording" % (recording_times.shape[0],len(recording_packets)))

x_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,0] )
y_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,1] )
z_interpolants = np.interp( playback_times, recording_times, recording_raceline[:,2] )
recording_interpolants = np.stack( (x_interpolants,y_interpolants,z_interpolants), axis=1 )

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(recording_interpolants[:,0], recording_interpolants[:,1], recording_interpolants[:,2], 'r')
plt.title('Recording Raceline')
plt.savefig('recording_raceline.png')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(playback_raceline[:,0], playback_raceline[:,1], playback_raceline[:,2], 'b')
plt.title('Playback Raceline')
plt.savefig('playback_raceline.png')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

diffs = playback_raceline-recording_interpolants
squared_diffs = np.square(diffs)
squared_norms = np.sum(squared_diffs,axis=1)
distances = np.sqrt(squared_norms)

fig3 = plt.figure()
plt.plot(playback_times, distances, 'b')
plt.title('Distance between recording and playback vs Time')
plt.xlabel("time (seconds)")
plt.ylabel("distance (meters???? I think?)")
plt.savefig('distances.png')
print("Mean Distance: %f" % (np.mean(distances)) )
#plt.axis([0, 6, 0, 20])
plt.show()


# with open(filename, 'r') as file:
#     jsonstring = file.read()
# data = TimestampedUDPData_pb2.TimestampedUDPData()
# google.protobuf.json_format.Parse(jsonstring,data)
# print(data)