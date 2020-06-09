def imageDataKey(data):
    return data.timestamp
def timestampedUdpPacketKey(packet):
    return packet.udp_packet.m_header.m_sessionTime