from posixpath import isabs
from typing import List
import os
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from .path_utils.pcd_utils import loadPCD

class TrackMap():
    def __init__(self, directory : str, align=False) -> None:
        self.starting_line_position : np.ndarray = None
        self.starting_line_rotation : Rotation = None
        self.length : float = None
        self.startinglinewidth : float = None
        self.inner_boundary : np.ndarray = None
        self.outer_boundary : np.ndarray = None
        self.raceline : np.ndarray = None
        self.directory : str = None
        self.name : str = None
        self.linemap : dict = dict()
        if directory is not None:
            self.loadFromDirectory(directory, align=align)
    def loadFromDirectory(self, directory : str, align=False):
        with open(os.path.join(directory, "metadata.yaml"), "r") as f:
            metadatadict : dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.starting_line_position = np.asarray(metadatadict["startingline_pose"]["position"], dtype=np.float64)
        self.starting_line_rotation = Rotation.from_quat(np.asarray(metadatadict["startingline_pose"]["quaternion"], dtype=np.float64))
        transform : np.ndarray = np.eye(4, dtype=np.float64)
        transform[0:3,0:3] = self.starting_line_rotation.inv().as_matrix()
        transform[0:3,3] = np.matmul(transform[0:3,0:3], -self.starting_line_position)
        self.startinglinewidth = metadatadict["startinglinewidth"]
        self.length = metadatadict["tracklength"]
        self.directory = directory
        self.name = os.path.basename(directory)
        for root, _, files in os.walk(directory, topdown = True):
            for name in files:
                base, ext = os.path.splitext(name)
                if ext==".pcd":
                    filepath = os.path.join(root, name)
                    _, line_track_, height, width = loadPCD(filepath, align=align)
                    line_track : np.ndarray = line_track_
                    line_track_x : np.ndarray = line_track['x'].copy()
                    line_track_y : np.ndarray = line_track['y'].copy()
                    line_track_z : np.ndarray = line_track['z'].copy()
                    line_track_ones : np.ndarray = np.ones_like(line_track_z)
                    line_track_all : np.ndarray = np.concatenate([line_track_x, line_track_y, line_track_z, line_track_ones], axis=1, dtype=line_track_x.dtype).T
                    line_map_all : np.ndarray = np.matmul(transform.astype(line_track_all.dtype), line_track_all)
                    line_map : np.ndarray = line_track.copy()
                    line_map["x"] = line_map_all[0].reshape(line_map["x"].shape)
                    line_map["y"] = line_map_all[1].reshape(line_map["y"].shape)
                    line_map["z"] = line_map_all[2].reshape(line_map["z"].shape)
                    self.linemap[base] = {"filepath" : filepath, "line" : line_map, "height" : height, "width": width}

        

def imageDataKey(data):
    return data.timestamp
def timestampedUdpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
trackNames=["Australia", "France", "China", "Bahrain", "Spain", "Monaco",\
            "Canada", "Britain", "Germany", "Hungary", "Belgium", "Italy",\
            "Singapore", "Japan", "Abu_Dhabi", "USA", "Brazil", "Austria",\
            "Russia", "Mexico", "Azerbaijan", "Bahrain_short", "Britan_short",\
            "USA_short", "Japan_short", "Rice242"]
def searchForFile(filename : str, searchdirs : List[str]):
    if os.path.isabs(filename):
        return filename
    for searchdir in searchdirs:
        if os.path.isdir(searchdir):
            entries = os.scandir(searchdir)
            for entry in entries:
                if os.path.isfile(entry.path) and entry.name==filename:
                    return entry.path
    return None 
def searchForTrackmap(trackname : str, searchdirs : List[str], align=False):
    for searchdir in searchdirs:
        for root, directories, _ in os.walk(searchdir, topdown = True):
            for directory in directories:
                full_directory = os.path.join(root,directory)
                print("searching %s in %s" % (directory,root))
                if directory==trackname and os.path.isfile(os.path.join(full_directory,"DEEPRACING_TRACKMAP")) and os.path.isfile(os.path.join(full_directory,"metadata.yaml")):
                    print("Yay!")
                    return TrackMap(os.path.join(root,directory), align=align)
    return None

class CarGeometry():
    def __init__(self, wheelbase : float = 3.698, length : float = 5.733, width : float = 2.0, tire_radius : float = .330, tire_width : float = .365):
        self.wheelbase : float = wheelbase
        self.length : float = length
        self.width : float = width
        self.tire_radius : float = tire_radius
        self.tire_width : float = tire_width
    def wheelPositions(self, dtype=np.float64):
        #Wheel order: RL, RR, FL, FR, returned as a homogenous matrix, each column is the position
        #of the corresponding wheel in a frame attached to the centroid of the chassis, with X pointing left and Z pointing forward
        rtn : np.ndarray = np.zeros((4, 4), dtype=dtype)
        rtn[0] = self.width/2.0 - self.tire_width/2.0
        rtn[0,[1,3]]*=-1.0
        rtn[2] = -self.length/2.0 + 2.0*self.tire_radius
        rtn[2,2:] += self.wheelbase
        rtn[3] = 1.0
        return rtn