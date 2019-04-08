import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

data = np.loadtxt("out.csv", delimiter=',')

steerleft = data[data[:,0]>=0]
steerright = data[data[:,0]<=0]


fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("vJoy vs Magic F1")    
ax1.set_xlabel('Magic F1 Value')
ax1.set_ylabel('vJoy Value')

ax1.plot(steerleft[:,0],steerleft[:,1], c='r', label='Left Steering')
ax1.plot(steerright[:,0],steerright[:,1], c='b', label='Right Steering')

steerleft_fit = np.polyfit(steerleft[:,0],steerleft[:,1],1)
steerright_fit = np.polyfit(steerright[:,0],steerright[:,1],1)
print("Left-steering equation: vjoy = %f*f1 + %f" % (steerleft_fit[0], steerleft_fit[1]))
print("Right-steering equation: vjoy = %f*f1 + %f" % (steerright_fit[0], steerright_fit[1]))
leg = ax1.legend()

plt.savefig("vjoy_calibration")
plt.show()