from posixpath import isabs
from typing import List
import os
import numpy as np

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