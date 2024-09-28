from posixpath import isabs
from typing import List, Union
import os
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from .path_utils.pcd_utils import loadPCD
from .path_utils import SmoothPathHelper
try:
    from .py_waymo_conversions import scenario_to_tfexample # type: ignore
except ImportError:
    pass
def structured_to_dense(structured : np.ndarray, keys=["x", "y", "z"]):
    return np.concatenate([structured[k] for k in keys], axis=1)
class TrackMap():
    def __init__(self, directory : str, align=False, transform_to_map = True) -> None:
        self.starting_line_position : np.ndarray = None
        self.starting_line_rotation : Rotation = None
        self.length : float = None
        self.startinglinewidth : float = None
        self.inner_boundary : np.ndarray = None
        self.outer_boundary : np.ndarray = None
        self.raceline : np.ndarray = None
        self.width_map : np.ndarray = None
        self.directory : str = None
        self.name : str = None
        self.linemap : dict[str,dict[str, np.ndarray]] = dict()
        self.frame_id : str = None
        self.clockwise : bool = None
        self.map_coords : bool = None
        self.align : bool = None
        if directory is not None:
            self.loadFromDirectory(directory, align=align, transform_to_map = transform_to_map)
            self.map_coords = transform_to_map

    def getPathHelper(self, key : str, dtype = np.float32, with_z = True) -> Union[SmoothPathHelper, None]:
        try:
            line_structured : np.ndarray = self.linemap[key]["line"]
            if with_z:
                line : np.ndarray = np.concatenate([line_structured["x"], line_structured["y"], line_structured["z"]], axis=1, dtype=dtype)
            else:
                line : np.ndarray = np.concatenate([line_structured["x"], line_structured["y"]], axis=1, dtype=dtype)
            if "speed" in line_structured.dtype.names:
                speeds : np.ndarray = np.squeeze(line_structured["speed"]).astype(dtype)
            else:
                speeds = None
            return SmoothPathHelper(line, speeds=speeds)
        except KeyError as e:
            return None
    def loadFromDirectory(self, directory : str, align=False, transform_to_map = True):
        with open(os.path.join(directory, "metadata.yaml"), "r") as f:
            metadatadict : dict = yaml.load(f, Loader=yaml.SafeLoader)
        self.align = align
        self.map_coords = transform_to_map
        self.clockwise = metadatadict["clockwise"]
        self.starting_line_position = np.asarray(metadatadict["startingline_pose"]["position"], dtype=np.float64)
        self.starting_line_rotation = Rotation.from_quat(np.asarray(metadatadict["startingline_pose"]["quaternion"], dtype=np.float64))
        transform : np.ndarray = np.eye(4, dtype=np.float64)
        transform[0:3,0:3] = self.starting_line_rotation.inv().as_matrix()
        transform[0:3,3] = np.matmul(transform[0:3,0:3], -self.starting_line_position)
        self.startinglinewidth = metadatadict["startinglinewidth"]
        self.length = metadatadict["tracklength"]
        self.directory = directory
        self.name = metadatadict["name"]
        if transform_to_map:
            self.frame_id = "map"
        else:
            self.frame_id = "track"
        for root, _, files in os.walk(directory, topdown = True):
            for name in files:
                base, ext = os.path.splitext(name)
                if ext==".pcd":
                    filepath = os.path.join(root, name)
                    _, line_track_, height, width = loadPCD(filepath, align=align)
                    line_track : np.ndarray = line_track_
                    if transform_to_map:
                        line_track_x : np.ndarray = line_track['x'][:,0].copy()
                        line_track_y : np.ndarray = line_track['y'][:,0].copy()
                        line_track_z : np.ndarray = line_track['z'][:,0].copy()  
                        line_track_ones : np.ndarray = np.ones_like(line_track_z)
                        line_track_all : np.ndarray = np.stack([line_track_x, line_track_y, line_track_z, line_track_ones], axis=0).astype(line_track_x.dtype)
                        line_map_all : np.ndarray = transform.astype(line_track_all.dtype) @ line_track_all
                        line_map : np.ndarray = line_track.copy()
                        line_map["x"] = line_map_all[[0,]].T
                        line_map["y"] = line_map_all[[1,]].T
                        line_map["z"] = line_map_all[[2,]].T
                        keyset : set = set(line_map.dtype.fields.keys())
                        quatkeys = ['i', 'j', 'k', 'w']
                        if keyset.intersection(set(quatkeys))==set(quatkeys):
                            quats_track  = np.concatenate([line_track[k] for k in quatkeys], axis=1)
                            rots_map = self.starting_line_rotation.inv() * Rotation.from_quat(quats_track)
                            quats_map = rots_map.as_quat()
                            quats_map[quats_track[:,-1]<0]*=-1.0
                            for (i, k) in enumerate(quatkeys):
                                line_map[k] = quats_map[:,[i,]]
                        self.linemap[base] = {"filepath" : filepath, "line" : line_map, "height" : height, "width": width}
                    else:
                        self.linemap[base] = {"filepath" : filepath, "line" : line_track, "height" : height, "width": width}
        self.inner_boundary = self.linemap["inner_boundary"]["line"]
        self.outer_boundary = self.linemap["outer_boundary"]["line"]
        self.raceline = self.linemap["raceline"]["line"]
        self.centerline = self.linemap["centerline"]["line"]
        width_map = self.linemap.get("widthmap")
        self.width_map = width_map["line"] if width_map is not None else None

        

def imageDataKey(data):
    return data.timestamp
def timestampedUdpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
trackNames={
    0 : "Australia",
    1 : "Canada",
    2 : "China",
    3 : "Bahrain",
    4 : "Catalunya",
    5 : "Monaco",
    6 : "Montreal",
    7 : "Britain",
    8 : "Hockenheim",
    9 : "Hungaroring",
    10 : "Spa",
    11 : "Monza",
    12 : "Singapore",
    13 : "Suzuka",
    14 : "AbuDhabi",
    15 : "TexasF1",
    16 : "Brazil",
    17 : "Austria",
    18 : "Sochi",
    19 : "Mexico",
    20 : "Azerbaijan",
    21 : "SakhirShort",
    22 : "BritainShort",
    23 : "TexasShort",
    24 : "SuzukaShort",
    25 : "Hanoi",
    26 : "Zandvoort",
    27 : "Imola",
    28 : "Portimao",
    29 : "Jeddah",
    30 : "Miami",
    31 : "VegasF1",
    32 : "Losail",
    33 : "VegasIndycar",
    34 : "TexasIndycar"
}
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

def searchForTrackmap(trackname : str, searchdirs : List[str], align=False, transform_to_map = True):
    for searchdir in searchdirs:
        for root, directories, _ in os.walk(searchdir, topdown = True):
            for directory in directories:
                full_directory = os.path.join(root,directory)
                print("searching %s in %s" % (directory,root))
                metadata_file = os.path.join(full_directory,"metadata.yaml")
                if not os.path.isfile(metadata_file):
                    continue
                with open(metadata_file, "r") as f:
                    metadata : dict = yaml.load(f, Loader=yaml.SafeLoader)
                if metadata.get("name", None)==trackname and os.path.isfile(os.path.join(full_directory,"DEEPRACING_TRACKMAP")):
                    print("Yay!")
                    return TrackMap(os.path.join(root,directory), align=align, transform_to_map = transform_to_map)
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