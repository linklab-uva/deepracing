import numpy as np
def loadArmaFile(filepath : str):
    matin = np.loadtxt(filepath,skiprows=2, delimiter="\t")
    t = matin[:,0]
    x = matin[:,1:4]
    xdot = matin[:,4:]
    return t, x, xdot
def writeArmaFile(filepath : str, t, x, xdot, delimiter="\t"):
    matout = np.hstack((np.array([t]).transpose(),x,xdot))
    headerstring = "ARMA_MAT_TXT_FN008\n" + \
                str(matout.shape[0]) + " " + str(matout.shape[1])+"\n"
    with open(filepath,"w") as f:
        f.write(headerstring)
        #linestrings = ["\t".join(map(str, matout[i].tolist()))+"\n" for i in range(matout.shape[0])]
        linestrings = [delimiter.join(format(x, "<E") for x in matout[i].tolist())+"\n" for i in range(matout.shape[0])]
        
        f.writelines(linestrings)