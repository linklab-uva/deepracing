import numpy as np
def loadArmaFile(filepath : str):
    matin = np.loadtxt(filepath,skiprows=2, delimiter="\t")
    t = matin[:,0]
    x = matin[:,1:4]
    xdot = matin[:,4:]
    return t, x, xdot
