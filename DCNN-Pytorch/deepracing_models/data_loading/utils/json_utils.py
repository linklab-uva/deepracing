
import os
import json
import numpy as np
import tqdm
import pickle as pkl
from scipy.spatial.transform import Rotation
def load_json_files(datadir: str) -> list[dict]:
    cachefile = os.path.join(datadir, "cache.pkl")
    if os.path.isfile(cachefile):
        with open(cachefile, 'rb') as f:
            json_data : list[dict] = pkl.load(f)
    else:
        json_data : list[dict] = []
        for filename in tqdm.tqdm(os.listdir(datadir), desc="Loading JSON files"):
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
    return json_data
def extract_header_data(json_data: list[dict], 
                        float_type=float, int_type=int) -> \
    tuple[np.ndarray, np.ndarray, np.ndarray]:
    numpackets = len(json_data)
    session_times = np.empty(numpackets, dtype=float_type)
    frame_identifiers = np.empty(numpackets, dtype=int_type)
    overall_frame_identifiers = np.empty(numpackets, dtype=int_type)
    player_car_indices = np.empty(numpackets, dtype=int_type)
    for i, packet in tqdm.tqdm(enumerate(json_data), desc="Extracting header data", total=numpackets):
        session_times[i] = packet["mHeader"]["mSessionTime"]
        frame_identifiers[i] = packet["mHeader"]["mFrameIdentifier"]
        player_car_indices[i] = packet["mHeader"]["mPlayerCarIndex"]
        overall_frame_identifiers[i] = packet["mHeader"].get("mOverallFrameIdentifier", frame_identifiers[i])
    return session_times, frame_identifiers, overall_frame_identifiers, player_car_indices

def numpify_motion_data(datadir : str, float_type=float, int_type=int) -> dict[str, np.ndarray]:

    json_data = load_json_files(datadir)
    numpackets = len(json_data)
    if numpackets == 0:
        return None
    numdrivers = len(json_data[0]["mCarMotionData"])

    session_times, frame_identifiers, overall_frame_identifiers, player_car_indices = extract_header_data(json_data, float_type, int_type)
    positions = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    quaternions = np.empty([numdrivers, numpackets, 4], dtype=float_type)
    velocities = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    accelerations = np.empty([numdrivers, numpackets, 3], dtype=float_type)
    for i, packet in tqdm.tqdm(enumerate(json_data), desc="Processing motion data packets", total=numpackets):
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
            rotmat = np.column_stack((forward_dir, left_dir, up_dir))
            quaternions[j, i] = Rotation.from_matrix(rotmat).as_quat()
            accelerations[j, i, 0] = motion_data["mGForceLongitudinal"]
            accelerations[j, i, 1] = motion_data["mGForceLateral"]
            accelerations[j, i, 2] = motion_data["mGForceVertical"]
            
    accelerations*=9.81
    return {"session_times" : session_times, "frame_identifiers" : frame_identifiers, "overall_frame_identifiers" : overall_frame_identifiers, "player_car_indices" : player_car_indices, \
           "positions" : positions, "quaternions" : quaternions, "velocities" : velocities, "accelerations" : accelerations}

def numpify_lap_data(datadir : str, float_type=float, int_type=int)-> dict[str, np.ndarray]:

    json_data = load_json_files(datadir)
    numpackets = len(json_data)
    if numpackets == 0:
        return None
    numdrivers = len(json_data[0]["mLapData"])

    session_times, frame_identifiers, overall_frame_identifiers, player_car_indices = extract_header_data(json_data, float_type, int_type)
    car_positions = np.empty([numdrivers, numpackets], dtype=float_type)
    lap_distances = np.empty([numdrivers, numpackets], dtype=float_type)
    total_distances = np.empty([numdrivers, numpackets], dtype=float_type)
    current_lap_times = np.empty([numdrivers, numpackets], dtype=float_type)
    last_lap_times = np.empty([numdrivers, numpackets], dtype=float_type)
    car_positions = np.empty([numdrivers, numpackets], dtype=int_type)
    lap_numbers = np.empty([numdrivers, numpackets], dtype=int_type)
    driver_statuses = np.empty([numdrivers, numpackets], dtype=int_type)
    result_statuses = np.empty([numdrivers, numpackets], dtype=int_type)
    pit_statuses = np.empty([numdrivers, numpackets], dtype=int_type)
    pit_lane_timer_active = np.empty([numdrivers, numpackets], dtype=int_type)

    for i, packet in tqdm.tqdm(enumerate(json_data), desc="Processing lap data packets", total=numpackets):
        lap_data = packet["mLapData"]
        for j in range(numdrivers):
            car_positions[j,i]= lap_data[j]["mCarPosition"]
            driver_statuses[j,i]= lap_data[j]["mDriverStatus"]
            result_statuses[j,i]= lap_data[j]["mResultStatus"]
            lap_distances[j,i]= lap_data[j]["mLapDistance"]
            total_distances[j,i]= lap_data[j]["mTotalDistance"]
            current_lap_times[j,i]= lap_data[j]["mCurrentLapTime"]
            last_lap_times[j,i]= lap_data[j]["mLastLapTime"]
            lap_numbers[j,i]= lap_data[j]["mCurrentLapNum"]
            pit_statuses[j,i]= lap_data[j]["mPitStatus"]
            pit_lane_timer_active[j,i]= int(
                (result_statuses[j,i]==2)*(( driver_statuses[j,i]==4) + ( driver_statuses[j,i]==3) + ( driver_statuses[j,i]==2) + ( driver_statuses[j,i]==1))
            )

    # accelerations*=9.81
    return {"session_times" : session_times, "frame_identifiers" : frame_identifiers, "overall_frame_identifiers" : overall_frame_identifiers, "player_car_indices" : player_car_indices,
            "car_positions" : car_positions, "driver_status" : driver_statuses, "result_status" : result_statuses,
            "lap_distances" : lap_distances, "total_distances" : total_distances, "current_lap_times" : current_lap_times,
            "last_lap_times" : last_lap_times, "lap_numbers" : lap_numbers, "pit_status" : pit_statuses, "pit_lane_timer_active" : pit_lane_timer_active}

def numpify_session_data(datadir : str, float_type=float, int_type=int) -> dict[str, np.ndarray]:
    json_data = load_json_files(datadir)
    numpackets = len(json_data)
    if numpackets == 0:
        return None
    track_ids = np.empty(numpackets, dtype=int_type)
    session_times, frame_identifiers, overall_frame_identifiers, player_car_indices = extract_header_data(json_data, float_type, int_type)
    for i, packet in tqdm.tqdm(enumerate(json_data), desc="Processing session data packets", total=numpackets):
        track_ids[i] = packet["mTrackId"]

    # accelerations*=9.81
    return {"session_times" : session_times, "frame_identifiers" : frame_identifiers, "overall_frame_identifiers" : overall_frame_identifiers, "player_car_indices" : player_car_indices, "track_ids" : track_ids}