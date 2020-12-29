import numpy as np
def writeArmaFile(fileout, t, X, Xdot):
    matout = np.hstack((np.array([t]).transpose(),X,Xdot))
    headerstring = "ARMA_MAT_TXT_FN008\n" + \
                    str(matout.shape[0]) + " " + str(matout.shape[1])
    np.savetxt(fileout, matout, delimiter="\t", header=headerstring, comments="")
def readArmaFile(filein):
    arrin = np.loadtxt(filein,skiprows=2,delimiter="\t")
    t = arrin[:,0]
    X = arrin[:,1:4]
    Xdot = arrin[:,4:]
    return t, X, Xdot
