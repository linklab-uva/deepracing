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
calibration_file = 'C:\\Users\\ttw2x\\Documents\\git_repos\\deepracing\\data-logger\\calibration_data.csv'
f = open(calibration_file)
dataz = np.genfromtxt(calibration_file, skip_header = 1, delimiter=",")
#print(dataz)
vjoy_angle = dataz[:,0].copy()/max(abs(dataz[:,0]))
telemetry_angle = dataz[:,1].copy()
physical_angle = dataz[:,2].copy()
positive_vjoy_angles = vjoy_angle[physical_angle>=0]
negative_vjoy_angles = vjoy_angle[physical_angle<0]
positive_telemetry_angles = telemetry_angle[physical_angle>=0]
negative_telemetry_angles = telemetry_angle[physical_angle<0]
positive_physical_angles = physical_angle[physical_angle>=0]
negative_physical_angles = physical_angle[physical_angle<0]
A = np.stack((np.ones_like(telemetry_angle), telemetry_angle), axis=1)
#print(telemetry_angle)
#print(physical_angle)
print(A)
print(A.shape)
X = positive_physical_angles
Y = positive_vjoy_angles
slope, intercept, r_value, p_value, std_err = stats.linregress( X , Y )
fit = np.array( ( slope , intercept ) )
print(fit)
print("R2 value: %f" % (r_value**2) )
print("Standard Error: %f" % (std_err) )
fitvals = np.polyval(fit,X)
fig = plt.figure("Data and fit line")
plt.plot(X, Y, label='data')
plt.plot(X, fitvals, label='best fit line')
fig.legend()
plt.show()
