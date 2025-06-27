
import os
import json
import numpy as np
import tqdm
import pickle as pkl
from scipy.spatial.transform import Rotation
def numpify_motion_data(datadir : str, float_type=float, int_type=int) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    cachefile = os.path.join(datadir, "cache.pkl")
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as f:
            json_data : list[dict] = pkl.load(f)
    else:
        json_data : list[dict] = []
        for filename in tqdm.tqdm(os.listdir(datadir), desc="Loading JSON files for motion data"):
            if filename.endswith('.json'):
                filepath = os.path.join(datadir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    json_data.append(data["udpPacket"])
        if len(json_data) == 0:
            return None
        def sortkey(packet : dict):
            return packet["mHeader"]["mSessionTime"]
        json_data.sort(key=sortkey)
        with open(cachefile, 'wb') as f:
            pkl.dump(json_data, f)

    numpackets = len(json_data)
    numdrivers = len(json_data[0]["mCarMotionData"])

    session_times = np.empty(numpackets, dtype=float_type)
    frame_identifiers = np.empty(numpackets, dtype=int_type)
    overall_frame_identifiers = np.empty(numpackets, dtype=int_type)
    positions = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    quaternions = np.empty([numdrivers, numpackets, 4], dtype=float_type)
    velocities = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    accelerations = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    for i, packet in tqdm.tqdm(enumerate(json_data), desc="Processing motion data packets", total=numpackets):
        session_times[i] = packet["mHeader"]["mSessionTime"]
        frame_identifiers[i] = packet["mHeader"]["mFrameIdentifier"]
        overall_frame_identifiers[i] = packet["mHeader"].get("mOverallFrameIdentifier", frame_identifiers[i])

        for j in range(numdrivers):
            motion_data = packet["mCarMotionData"][j]
            positions[j, i] = np.asarray([motion_data["mWorldPosition%s" % s] for s in "XYZ"], dtype=float_type)
            velocities[j, i] = np.asarray([motion_data["mWorldVelocity%s" % s] for s in "XYZ"], dtype=float_type)
            forward_dir = np.asarray([motion_data["mWorldForwardDir%s" % s] for s in "XYZ"], dtype=float_type)
            forward_dir = forward_dir / np.linalg.norm(forward_dir, ord=2.0, axis=0)
            left_dir = -np.asarray([motion_data["mWorldRightDir%s" % s] for s in "XYZ"], dtype=float_type)
            left_dir = left_dir / np.linalg.norm(left_dir, ord=2.0, axis=0)
            up_dir = np.cross(forward_dir, left_dir)
            up_dir = up_dir / np.linalg.norm(up_dir, ord=2.0, axis=0)
            rotmat = np.column_stack((left_dir, up_dir, forward_dir))
            quaternions[j, i] = Rotation.from_matrix(rotmat).as_quat()
            accelerations[j, i, 0] = motion_data["mGForceLongitudinal"]
            accelerations[j, i, 1] = motion_data["mGForceLateral"]
            accelerations[j, i, 2] = motion_data["mGForceVertical"]
            
    accelerations*=9.81
    return session_times, frame_identifiers, overall_frame_identifiers, \
           positions, quaternions, velocities, accelerations