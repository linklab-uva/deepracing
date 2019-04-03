import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

data = np.loadtxt("out.csv", delimiter=',')

x = data[:,0]
y = data[:,1]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Magic F1 vs vJoy")    
ax1.set_xlabel('vJoy Value')
ax1.set_ylabel('Magic F1 Value')

ax1.plot(x,y, c='r', label='the curve')

leg = ax1.legend()

plt.savefig("vjoy_calibration")
plt.show()