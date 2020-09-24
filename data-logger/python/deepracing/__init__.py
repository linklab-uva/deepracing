from typing import List
import os
def imageDataKey(data):
    return data.timestamp
def timestampedUdpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime
trackNames=["Australia", "France", "China", "Bahrain", "Spain", "Monaco",\
            "Canada", "Britain", "Germany", "Hungary", "Belgium", "Italy",\
            "Singapore", "Japan", "Abu_Dhabi", "USA", "Brazil", "Austria",\
            "Russia", "Mexico", "Azerbaijan", "Bahrain_short", "Britan_short",\
            "USA_short", "Japan_short"]
def searchForFile(filename : str, searchdirs : List[str]):
    for searchdir in searchdirs:
        entries = os.scandir(searchdir)
        for entry in entries:
            if entry.name == filename
                return entry.path
    return None 