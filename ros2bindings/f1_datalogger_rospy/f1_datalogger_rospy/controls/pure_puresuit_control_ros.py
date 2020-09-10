import cv2
import numpy as np
import argparse
import os
import time
import logging
import cv2
from numpy_ringbuffer import RingBuffer as RB
import yaml
import torch
import torchvision
import torchvision.transforms as tf
import deepracing.imutils
import scipy
import scipy.interpolate
import py_f1_interface
import deepracing.pose_utils
import deepracing
import threading
import numpy.linalg as la
import scipy.integrate as integrate
import socket
import scipy.spatial
import queue
import deepracing_models.math_utils as mu
import torch
import torch.nn as NN
import torch.utils.data as data_utils
import deepracing_models.nn_models.Models
import matplotlib.pyplot as plt
from f1_datalogger_msgs.msg import BoundaryLine, TimestampedPacketCarStatusData, TimestampedPacketCarTelemetryData, TimestampedPacketMotionData, PacketCarTelemetryData, PacketMotionData, CarMotionData, CarStatusData, CarTelemetryData, PacketHeader
from geometry_msgs.msg import Vector3Stamped, Vector3, PointStamped, Point, PoseStamped, Pose, Quaternion, PoseArray
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Float64
import rclpy
from rclpy.node import Node
from rclpy import Parameter
from copy import deepcopy
import sensor_msgs
from scipy.spatial.kdtree import KDTree
from shapely.geometry import Point as ShapelyPoint, MultiPoint#, Point2d as ShapelyPoint2d
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing

import timeit
class PurePursuitControllerROS(Node):
    def __init__(self):
        super(PurePursuitControllerROS,self).__init__('pure_pursuit_control', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.packet_queue = queue.Queue()
        self.running = True
        self.current_motion_packet : TimestampedPacketMotionData  = TimestampedPacketMotionData()
        self.current_motion_data : CarMotionData  = CarMotionData()
        self.current_status_data : CarStatusData  = CarStatusData()
        self.current_telemetry_data : CarTelemetryData  = CarTelemetryData()
        self.setpoint_publisher = self.create_publisher(Float64, "vel_setpoint", 1)
        self.dt_publisher = self.create_publisher(Float64, "dt", 1)
        self.velsetpoint = 0.0
        self.current_speed = 0.0
        self.throttle_out = 0.0
        self.controller = py_f1_interface.F1Interface(1)
        self.controller.setControl(0.0,0.0,0.0)

        


        gpu_param : Parameter = self.get_parameter_or("gpu",Parameter("gpu", value=0))
        self.gpu : int = gpu_param.get_parameter_value().integer_value

        max_speed_param : Parameter = self.get_parameter_or("max_speed", Parameter("max_speed",value=200.0))
        self.max_speed : float = max_speed_param.get_parameter_value().double_value
        
        max_centripetal_acceleration_param : Parameter = self.get_parameter_or("max_centripetal_acceleration", Parameter("max_centripetal_acceleration",value=20.0))
        self.max_centripetal_acceleration : float = max_centripetal_acceleration_param.get_parameter_value().double_value

        
        L_param : Parameter = self.get_parameter_or("wheelbase",Parameter("wheelbase", value=3.5))
        self.L = float = L_param.get_parameter_value().double_value

        lookahead_gain_param : Parameter = self.get_parameter_or("lookahead_gain", Parameter("lookahead_gain",value=0.25))
        self.lookahead_gain : float = lookahead_gain_param.get_parameter_value().double_value

        velocity_lookahead_gain_param : Parameter = self.get_parameter_or("velocity_lookahead_gain", Parameter("velocity_lookahead_gain",value=0.25))
        self.velocity_lookahead_gain : float = velocity_lookahead_gain_param.get_parameter_value().double_value
        
        left_steer_factor_param : Parameter = self.get_parameter_or("left_steer_factor", Parameter("left_steer_factor",value=3.39814))
        self.left_steer_factor : float = left_steer_factor_param.get_parameter_value().double_value
        
        left_steer_offset_param : Parameter = self.get_parameter_or("left_steer_offset", Parameter("left_steer_offset",value=0.0))
        self.left_steer_offset : float = left_steer_offset_param.get_parameter_value().double_value
        
        right_steer_factor_param : Parameter = self.get_parameter_or("right_steer_factor", Parameter("right_steer_factor",value=3.72814))
        self.right_steer_factor : float = right_steer_factor_param.get_parameter_value().double_value
        
        right_steer_offset_param : Parameter = self.get_parameter_or("right_steer_offset", Parameter("right_steer_offset",value=0.0))
        self.right_steer_offset : float = right_steer_offset_param.get_parameter_value().double_value
        
        use_drs_param : Parameter = self.get_parameter_or("use_drs", Parameter("use_drs",value=False))
        self.use_drs : bool = use_drs_param.get_parameter_value().bool_value

        boundary_check_param : Parameter = self.get_parameter_or("boundary_check", Parameter("boundary_check",value=False))
        self.boundary_check : bool = boundary_check_param.get_parameter_value().bool_value

        
        if self.use_drs:
            print("Using DRS")
        else:
            print("Not using DRS")

        self.inner_boundary = None
        self.inner_boundary_inv = None
        self.inner_boundary_kdtree = None
        # self.inner_boundary_normals = None

        self.outer_boundary = None
        self.outer_boundary_inv = None
        self.outer_boundary_kdtree = None
        # self.outer_boundary_tangents = None
        # self.outer_boundary_normals = None
        self.track_distance = 5303.0



        

        self.current_pose : PoseStamped = PoseStamped()
        self.current_pose_mat = torch.zeros([4,4],dtype=torch.float64)#.cuda(self.gpu)
        self.current_pose_mat[3,3]=1.0
        self.current_pose_inv_mat = self.current_pose_mat.clone()
        self.pose_sub = self.create_subscription( PoseStamped, '/car_pose', self.poseCallback, 1)

        self.motion_data_sub = self.create_subscription(
            TimestampedPacketMotionData,
            '/motion_data',
            self.velocityControl,
            1)
        self.status_data_sub = self.create_subscription(
            TimestampedPacketCarStatusData,
            '/status_data',
            self.statusUpdate,
            1)
        self.telemetry_data_sub = self.create_subscription(
            TimestampedPacketCarTelemetryData,
            '/telemetry_data',
            self.telemetryUpdate,
            1)
        if self.boundary_check:
            self.inner_boundary_sub = self.create_subscription(
                PoseArray,
                '/inner_track_boundary/pose_array',
                self.innerBoundaryCB,
                1)
            self.outer_boundary_sub = self.create_subscription(
                PoseArray,
                '/outer_track_boundary/pose_array',
                self.outerBoundaryCB,
                1)
        # self.racingline_sub = self.create_subscription(
        #     PoseArray,
        #     '/optimal_raceline/pose_array',
        #     self.racelineCB,
        #     1)
        self.control_thread = threading.Thread(target=self.lateralControl)
        
    def innerBoundaryCB(self, boundary_msg: PoseArray ):
        if self.boundary_check and (self.inner_boundary is None):
            positions = np.row_stack([np.array([p.position.x, p.position.y, p.position.z]) for p in boundary_msg.poses])
            self.inner_boundary_kdtree = KDTree(positions)
            quaternions = np.row_stack([np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in boundary_msg.poses])
            rotations = Rot.from_quat(quaternions)
            rotmats = rotations.as_matrix()
            inner_boundary = torch.zeros(positions.shape[0], 4, 4, dtype=torch.float64, device=torch.device("cuda:%d"%self.gpu))
            inner_boundary[:,0:3,0:3] = torch.from_numpy(rotmats).double().cuda(self.gpu)
            inner_boundary[:,0:3,3] = torch.from_numpy(positions).double().cuda(self.gpu)
            inner_boundary[:,3,3]=1.0
            inner_boundary_inv = torch.inverse(inner_boundary)
          #  print(inner_boundary[0:10])
            self.inner_boundary, self.inner_boundary_inv = (inner_boundary, inner_boundary_inv)
            del self.inner_boundary_sub
            
    def outerBoundaryCB(self, boundary_msg: PoseArray ):
        if self.boundary_check and (self.outer_boundary is None):
            positions = np.row_stack([np.array([p.position.x, p.position.y, p.position.z]) for p in boundary_msg.poses])
            self.outer_boundary_kdtree = KDTree(positions)
            quaternions = np.row_stack([np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]) for p in boundary_msg.poses])
            rotations = Rot.from_quat(quaternions)
            rotmats = rotations.as_matrix()
            outer_boundary = torch.zeros(positions.shape[0], 4, 4, dtype=torch.float64, device=torch.device("cuda:%d"%self.gpu))
            outer_boundary[:,0:3,0:3] = torch.from_numpy(rotmats).double().cuda(self.gpu)
            outer_boundary[:,0:3,3] = torch.from_numpy(positions).double().cuda(self.gpu)
            outer_boundary[:,3,3]=1.0
            outer_boundary_inv = torch.inverse(outer_boundary)
            self.outer_boundary, self.outer_boundary_inv = (outer_boundary, outer_boundary_inv)
            del self.outer_boundary_sub
            
    def racelineCB(self, boundary_msg: BoundaryLine ):
        pass

    def poseCallback(self, pose_msg : PoseStamped):
        self.current_pose = pose_msg
        R = torch.from_numpy(Rot.from_quat( np.array([pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w], dtype=np.float64) ).as_matrix())
        v = torch.from_numpy(np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z], dtype=np.float64 ) )
        p = torch.cat([R, v.unsqueeze(1)], dim=1 )
        pinv = torch.cat([R.transpose(0,1), -torch.matmul(R.transpose(0,1),v).unsqueeze(1)], dim=1 )
        self.current_pose_mat[0:3], self.current_pose_inv_mat[0:3] = (p, pinv)#.cuda(self.gpu)
        #self.current_pose_inv_mat = torch.inverse(self.current_pose_mat)
        
    def start(self):
        self.control_thread.start()
    def stop(self):
        self.running = False
        time.sleep(0.5)

    def telemetryUpdate(self, msg : TimestampedPacketCarTelemetryData):
        self.current_telemetry_data = msg.udp_packet.car_telemetry_data[0]

    def statusUpdate(self, msg : TimestampedPacketCarStatusData):
        self.current_status_data = msg.udp_packet.car_status_data[0]

    def velocityControl(self, msg : TimestampedPacketMotionData):
        self.current_motion_packet = deepcopy(msg)
        packet : PacketMotionData = self.current_motion_packet.udp_packet
        header : PacketHeader = packet.header
        motion_data_vec : list = packet.car_motion_data
        if len(motion_data_vec)==0:
            return
        self.current_motion_data = motion_data_vec[header.player_car_index]
        velrosstamped : Vector3Stamped = self.current_motion_data.world_velocity
        if (velrosstamped.header.frame_id == ""):
           return
        velros : Vector3 = velrosstamped.vector
        vel = np.array( (velros.x, velros.y, velros.z), dtype=np.float64)
        speed = la.norm(vel)
        self.current_speed = speed
        
    def getTrajectory(self):
        return None, None, None
    def setControl(self):
        lookahead_positions, v_local_forward_, distances_forward_, = self.getTrajectory()
        if lookahead_positions is None:
            return
        if distances_forward_ is None:
            distances_forward = la.norm(lookahead_positions, axis=1)
        else:
            distances_forward = distances_forward_
        if v_local_forward_ is None:
            s = np.linspace(0.0,1.0,num=lookahead_positions.shape[0])
            posspline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(s, lookahead_positions.cpu().numpy(), k=3)
            tangentspline : scipy.interpolate.BSpline = posspline.derivative(nu=1)
            tangents = tangentspline(s)
            tangentnorms = np.linalg.norm(tangents,axis=1)
            normalspline : scipy.interpolate.BSpline = posspline.derivative(nu=2)
            normals = normalspline(s)
            crossproductnorms = tangents[:,0]*normals[:,1] - tangents[:,1]*normals[:,0]
            curvatures = crossproductnorms/np.power(tangentnorms, 3)
            radii = np.abs(1.0/(curvatures+1E-12))
            speeds = self.max_speed*np.ones(lookahead_positions.shape[0])
            centripetal_accelerations = np.power(speeds,2.0)/radii
            max_allowable_speeds = np.sqrt(self.max_centripetal_acceleration*radii)
            idx = centripetal_accelerations>self.max_centripetal_acceleration
            speeds[idx] = max_allowable_speeds[idx]
        else:
            speeds = torch.norm(v_local_forward_, p=2, dim=1)
        #lookahead_angles = np.arctan2(lookahead_positions[:,1], lookahead_positions[:,0])
        # velrosstamped : Vector3Stamped = deepcopy(self.current_motion_data.world_velocity)
        # if (velrosstamped.header.frame_id == ""):
        #     return
        # velros : Vector3 = velrosstamped.vector
        # vel = np.array( (velros.x, velros.y, velros.z), dtype=np.float64)
        # speed = la.norm(vel)
        lookahead_distance = max(self.lookahead_gain*self.current_speed, 5.0)
        lookahead_distance_vel = self.velocity_lookahead_gain*self.current_speed

        lookahead_index = torch.argmin(torch.abs(distances_forward-lookahead_distance))
        lookahead_index_vel = torch.argmin(torch.abs(distances_forward-lookahead_distance_vel))

        lookaheadVector = lookahead_positions[lookahead_index]
        lookaheadVectorVel = lookahead_positions[lookahead_index_vel]


        D = torch.norm(lookaheadVector, p=2)
        lookaheadDirection = lookaheadVector/D
        alpha = torch.atan2(lookaheadDirection[0],lookaheadDirection[1])
        physical_angle = (torch.atan((2 * self.L*torch.sin(alpha)) / D)).item()
        if (physical_angle > 0) :
            delta = self.left_steer_factor*physical_angle + self.left_steer_offset
        else:
            delta = self.right_steer_factor*physical_angle + self.right_steer_offset
        self.velsetpoint = speeds[lookahead_index_vel].item()
        self.setpoint_publisher.publish(Float64(data=self.velsetpoint))

        if self.velsetpoint>self.current_speed:
            self.controller.setControl(delta,1.0,0.0)
        else:
            self.controller.setControl(delta,0.0,1.0)
        # if self.use_drs and self.current_status_data.m_drs_allowed==1 and self.current_telemetry_data.drs==0:
        #     self.controller.pushDRS()
    def lateralControl(self):
        timer = timeit.Timer(stmt=self.setControl, timer=timeit.default_timer)
        while self.running:
            dt = timer.timeit(number=1)
            self.dt_publisher.publish(Float64(data=dt))
          #  self.setControl()
            # t1 = timeit.default_timer()
            # t2 = timeit.default_timer()
            # dt = t2 - t1