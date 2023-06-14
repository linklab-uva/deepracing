import numpy as np

def numpyToPCD(x : np.ndarray, points : np.ndarray, filepath : str, x_name : str = "time"):
    with open(filepath, "w") as f:
        headerlines = ["VERSION 0.7\n", 
                       "FIELDS x y z %s\n" % (x_name,), 
                       "SIZE 4 4 4 4\n", 
                       "TYPE F F F F\n", 
                       "COUNT 1 1 1 1\n", 
                       "WIDTH %d\n" % (points.shape[0],),
                       "HEIGHT 1\n",
                       "VIEWPOINT 0 0 0 1 0 0 0\n",
                       "POINTS %d\n" % (points.shape[0],),
                       "DATA ascii\n"]
        f.writelines(headerlines)
        block = np.concatenate([points, x[:,None]], axis=1)
        np.savetxt(f, block, fmt="%.5f")
