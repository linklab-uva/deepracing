import numpy as np
import numpy.linalg as la
import quaternion
import math
import torch
import deepracing.pose_utils
def quaternionDistanceTorch(q1, q2):
    return 2.0*torch.acos(torch.abs(torch.dot(q1, q2)))
def quaternionDistance(q1, q2):
    return 2.0*np.arccos(np.abs(np.dot(quaternion.as_float_array(q1), quaternion.as_float_array(q2))))
print("hi there")
v1 = np.random.rand(3)
v1 = v1/la.norm(v1)
if(np.random.rand()>0.5):
    v1 = -v1
angle1 = 2.0*math.pi*(np.random.rand())
q1 = quaternion.from_rotation_vector(angle1*v1)
print(q1)

v2 = np.random.rand(3)
v2 = v2/la.norm(v2)
if(np.random.rand()>0.5):
    v2 = -v2
angle2 = math.pi*(np.random.rand())
q2 = quaternion.from_rotation_vector(angle2*v2)
print(q2)

q1torch = torch.as_tensor(quaternion.as_float_array(q1))
print(q1torch)
q2torch = torch.as_tensor(quaternion.as_float_array(q2))
print(q2torch)

print("Distance between q1 and q2: %f" %(quaternionDistance(q1,q2)))
print("Distance between q1 and -q2: %f" %(quaternionDistance(q1,-q2)))

print("Distance between q1 and q2 torch: %f" %(quaternionDistanceTorch(q1torch,q2torch)))
print("Distance between q1 and -q2 torch: %f" %(quaternionDistanceTorch(q1torch,-q2torch)))


pose = np.eye(4,4)
pose[0:3,3] = np.random.rand(3)
pose[0:3,0:3] = quaternion.as_rotation_matrix(q1)
print(pose)
print(la.inv(pose))
print(deepracing.pose_utils.inverseTransform(pose))


