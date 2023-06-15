import numpy as np
from scipy.spatial.transform import Rotation

def numpyToPCD(x : np.ndarray, points : np.ndarray, filepath : str, 
               x_name : str = "time", viewpoint_pos : np.ndarray = np.zeros(3), viewpoint_rot : Rotation = Rotation.identity()):

    viewpoint_pose = np.concatenate([viewpoint_rot.as_quat(), viewpoint_pos])
    with open(filepath, "w") as f:
        headerlines = ["VERSION 0.7\n", 
                       "FIELDS x y z %s\n" % (x_name,), 
                       "SIZE 4 4 4 4\n", 
                       "TYPE F F F F\n", 
                       "COUNT 1 1 1 1\n", 
                       "WIDTH %d\n" % (points.shape[0],),
                       "HEIGHT 1\n",
                       "VIEWPOINT %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n" % tuple(viewpoint_pose.tolist()),
                       "POINTS %d\n" % (points.shape[0],),
                       "DATA ascii\n"]
        f.writelines(headerlines)
        block = np.concatenate([points, x[:,None]], axis=1)
        np.savetxt(f, block, fmt="%.5f")
