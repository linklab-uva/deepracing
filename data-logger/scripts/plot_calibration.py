import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',')
    return data

data = read_datafile('out.csv')

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