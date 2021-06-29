import TimestampedPacketMotionData_pb2
import TimestampedPacketLapData_pb2
import TimestampedPacketSessionData_pb2
import argparse
import os
import google.protobuf.json_format
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.transform import resize
from tqdm import tqdm as tqdm
from deepracing import protobuf_utils
from deepracing.imutils import resizeImage as resizeImage
from deepracing.imutils import readImage as readImage
from deepracing.protobuf_utils import getAllMotionPackets, getAllLapDataPackets, extractPosition, extractRotation, extractVelocity
from deepracing.backend import MultiAgentLabelLMDBWrapper, ImageLMDBWrapper, LaserScanLMDBWrapper
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RotationSpline as RotSpline
from scipy.interpolate import BSpline, make_interp_spline, make_lsq_spline
from Pose3d_pb2 import Pose3d
from typing import List, Tuple, Sequence
import torchvision.transforms as T
import torchvision.transforms.functional as F
from deepracing_models.data_loading.image_transforms import IdentifyTransform
import json
import yaml
from deepracing_models.data_loading import TimeIndex
import torch.distributions as dist

def packetKey(packet) -> float:
    return packet.udp_packet.m_header.m_sessionTime
class PoseVelocityDataset(Dataset):
    def __init__(self, dataset_root_directory : str, context_indices : int = 5, context_time : float = 1.75, prediction_time : float = 1.75, fake_length=1000):
        super(PoseVelocityDataset, self).__init__()      
        self.dataset_root_directory = dataset_root_directory
        self.udp_data_dir = os.path.join(self.dataset_root_directory, "udp_data")
        self.motion_data_dir = os.path.join(self.udp_data_dir, "motion_packets")
        self.lap_data_dir = os.path.join(self.udp_data_dir, "lap_packets")
        self.cache_dir = os.path.join(dataset_root_directory, "__pose_vel_cache__")
        self.context_indices = context_indices
        self.context_time = context_time
        self.prediction_time = prediction_time
        with open(os.path.join(self.dataset_root_directory, "f1_dataset_config.yaml"), "r") as f:
            self.dataset_config = yaml.load(f, Loader=yaml.SafeLoader)

        if os.path.isdir(self.cache_dir):
            motion_packet_times = np.load(os.path.join(self.cache_dir, "motion_packet_times.npy"))
            all_positions = np.load(os.path.join(self.cache_dir, "positions.npy"))
            all_velocities = np.load(os.path.join(self.cache_dir, "velocities.npy"))
            all_quaternions = np.load(os.path.join(self.cache_dir, "quaternions.npy"))
            
            lap_packet_times = np.load(os.path.join(self.cache_dir, "lap_packet_times.npy"))
            result_statuses = np.load(os.path.join(self.cache_dir, "result_statuses.npy"))
            
            with open(os.path.join(self.cache_dir, "metadata.yaml"), "r") as f:
                metadatadict = yaml.load(f, Loader=yaml.SafeLoader)
                player_car_idx = metadatadict["player_car_index"]
            assert(np.all(motion_packet_times==motion_packet_times))
            assert(np.all(all_positions==all_positions))
            assert(np.all(all_velocities==all_velocities))
            assert(np.all(all_quaternions==all_quaternions))
            assert(np.all(lap_packet_times==lap_packet_times))
            assert(np.all(result_statuses==result_statuses))
        else:
            all_motion_packets = sorted(getAllMotionPackets(self.motion_data_dir, self.dataset_config["use_json"]), key=packetKey)
            all_positions = np.nan*np.zeros((len(all_motion_packets), 20, 3), dtype=np.float64)
            all_velocities = np.nan*np.zeros((len(all_motion_packets), 20, 3), dtype=np.float64)
            all_quaternions = np.nan*np.zeros((len(all_motion_packets), 20, 4), dtype=np.float64)
            motion_packet_times = np.empty(len(all_motion_packets), dtype=np.float64)
            for i in tqdm(range(len(all_motion_packets)), desc="Extracting positions, velocities, and quaternions"):
                packet = all_motion_packets[i]
                all_positions[i] = np.stack([extractPosition(packet.udp_packet, car_index=j) for j in range(20)])
                all_velocities[i] = np.stack([extractVelocity(packet.udp_packet, car_index=j) for j in range(20)])
                all_quaternions[i] = np.stack([extractRotation(packet.udp_packet, car_index=j) for j in range(20)])
                motion_packet_times[i] = packetKey(packet)
            assert(np.all(all_positions==all_positions))
            assert(np.all(all_velocities==all_velocities))
            assert(np.all(all_quaternions==all_quaternions))

            all_lap_packets = sorted(getAllLapDataPackets(self.lap_data_dir, self.dataset_config["use_json"]), key=packetKey)
            lap_packet_times = np.asarray([packetKey(all_lap_packets[i]) for i in range(len(all_lap_packets))], dtype=np.float64)
            result_statuses = np.zeros((len(all_lap_packets), 20), dtype=np.int32)
            for i in tqdm(range(len(all_lap_packets)), desc="Extracting lap information"):
                lap_packet = all_lap_packets[i]
                result_statuses[i] = np.asarray([lap_packet.udp_packet.m_lapData[j].m_resultStatus for j in range(20)], dtype=np.int32)
            assert(np.all(lap_packet_times==lap_packet_times))
            assert(np.all(result_statuses==result_statuses))

            os.makedirs(self.cache_dir)
            player_car_idx = all_motion_packets[0].udp_packet.m_header.m_playerCarIndex
            with open(os.path.join(self.cache_dir, "metadata.yaml"), "w") as f:
                yaml.dump({"player_car_index" : player_car_idx}, f, Dumper=yaml.SafeDumper)
            np.save(os.path.join(self.cache_dir, "motion_packet_times.npy"), motion_packet_times)
            np.save(os.path.join(self.cache_dir, "positions.npy"), all_positions)
            np.save(os.path.join(self.cache_dir, "velocities.npy"), all_velocities)
            np.save(os.path.join(self.cache_dir, "quaternions.npy"), all_quaternions)
            np.save(os.path.join(self.cache_dir, "lap_packet_times.npy"), lap_packet_times)
            np.save(os.path.join(self.cache_dir, "result_statuses.npy"), result_statuses)


        self.position_splines : List[BSpline] = [make_interp_spline(motion_packet_times, all_positions[:,i]) for i in range(all_positions.shape[1])]
        self.velocity_splines : List[BSpline] = [make_interp_spline(motion_packet_times, all_velocities[:,i]) for i in range(all_velocities.shape[1])]
        self.quaternion_splines : List[RotSpline] = [RotSpline(motion_packet_times, Rot.from_quat(all_quaternions[:,i])) for i in range(all_quaternions.shape[1])]

        # Iclip = (motion_packet_times>(lap_packet_times[0] + 10.0))*(motion_packet_times<(lap_packet_times[-1] - 10.0))
        Iclip = (motion_packet_times>(lap_packet_times[0] + context_time + 0.5))*(motion_packet_times<(lap_packet_times[-1] - prediction_time - 0.5))
        self.all_positions=all_positions[Iclip]
        self.all_velocities=all_velocities[Iclip]
        self.all_quaternions=all_quaternions[Iclip]
        self.motion_packet_times=motion_packet_times[Iclip]
        assert(self.motion_packet_times.shape[0] == self.all_positions.shape[0] == self.all_velocities.shape[0] == self.all_quaternions.shape[0])
        
        self.time_dist = dist.Uniform(self.motion_packet_times[0], self.motion_packet_times[-1])

        self.fake_length = fake_length

        
        self.lap_packet_times=lap_packet_times
        self.result_statuses=result_statuses
        
        self.player_car_idx=player_car_idx

        self.lap_index : TimeIndex = TimeIndex(self.lap_packet_times, self.result_statuses)
        
    def __len__(self):
        return self.fake_length

    def __getitem__(self, idx):
        current_packet_time = self.time_dist.sample().item()
        _, statuses = self.lap_index.sample(current_packet_time - (self.context_time+0.25), current_packet_time + (self.prediction_time+0.25))
        valid = ~((statuses==0) + (statuses==1))
        allvalid = np.prod(valid, axis=0)
        allvalid[self.player_car_idx]=0
        valid_mask = allvalid.astype(np.bool8)

        tpast = np.linspace(current_packet_time - self.context_time, current_packet_time, self.context_indices)
        past_positions = np.empty((20, tpast.shape[0], 3), dtype=np.float64)
        past_velocities = np.empty((20, tpast.shape[0], 3), dtype=np.float64)
        past_quaternions = np.empty((20, tpast.shape[0], 4), dtype=np.float64)

        tfuture = np.linspace(current_packet_time, current_packet_time + self.prediction_time, 6*tpast.shape[0])
        future_positions = np.empty((past_positions.shape[0], tfuture.shape[0], 3), dtype=np.float64)
        future_velocities = np.empty((past_positions.shape[0], tfuture.shape[0], 3), dtype=np.float64)
        future_quaternions = np.empty((past_positions.shape[0], tfuture.shape[0], 4), dtype=np.float64)
        for i in range(past_positions.shape[0]):
            past_positions[i] = self.position_splines[i](tpast)
            past_velocities[i] = self.velocity_splines[i](tpast)
            q = self.quaternion_splines[i](tpast).as_quat()
            for j in range(1, q.shape[0]):
                if np.linalg.norm((-q[j]) - q[j-1])<np.linalg.norm(q[j] - q[j-1]):
                    q[j]*=-1.0
            past_quaternions[i] = q
            future_positions[i] = self.position_splines[i](tfuture)
            future_velocities[i] = self.velocity_splines[i](tfuture)
            qfuture = self.quaternion_splines[i](tfuture).as_quat()
            for j in range(1, qfuture.shape[0]):
                if np.linalg.norm((-qfuture[j]) - qfuture[j-1])<np.linalg.norm(qfuture[j] - qfuture[j-1]):
                    qfuture[j]*=-1.0
            future_quaternions[i] = qfuture
        ego_idx = self.player_car_idx
        tpast_ = np.stack([tpast.copy() for asdf in range(past_positions.shape[0])])
        tfuture_ = np.stack([tfuture.copy() for asdf in range(past_positions.shape[0])])

        


        return {"tpast": tpast_, "tfuture": tfuture_, "ego_idx": ego_idx, "current_packet_time" : current_packet_time, "valid_mask" : valid_mask, "past_positions" : past_positions, "past_velocities" : past_velocities, "past_quaternions": past_quaternions, "future_positions" : future_positions, "future_velocities": future_velocities, "future_quaternions": future_quaternions}
    
