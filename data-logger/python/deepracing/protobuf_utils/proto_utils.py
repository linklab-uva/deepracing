import TimestampedPacketSessionData_pb2
import TimestampedPacketCarTelemetryData_pb2

import os
import google.protobuf.json_format
def getAllTelemetryPackets(telemetry_folder: str, use_json: bool):
   telemetry_packets = []
   if use_json:
      filepaths = [os.path.join(telemetry_folder, f) for f in os.listdir(telemetry_folder) if os.path.isfile(os.path.join(telemetry_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in jsonstrings:
         data = TimestampedPacketCarTelemetryData_pb2.TimestampedPacketCarTelemetryData()
         google.protobuf.json_format.Parse(jsonstring, data)
         telemetry_packets.append(data)
   else:
      filepaths = [os.path.join(telemetry_folder, f) for f in os.listdir(telemetry_folder) if os.path.isfile(os.path.join(telemetry_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
         try:
            data = TimestampedPacketCarTelemetryData_pb2.TimestampedPacketCarTelemetryData()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            telemetry_packets.append(data)
         except:
            f.close()
            print("Could not read telemetry packet file %s." %(filepath))
            continue
   return telemetry_packets
def getAllSessionPackets(session_folder: str, use_json: bool):
   session_packets = []
   if use_json:
      filepaths = [os.path.join(session_folder, f) for f in os.listdir(session_folder) if os.path.isfile(os.path.join(session_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in jsonstrings:
         data = TimestampedPacketSessionData_pb2.TimestampedPacketSessionData()
         google.protobuf.json_format.Parse(jsonstring, data)
         session_packets.append(data)
   else:
      filepaths = [os.path.join(session_folder, f) for f in os.listdir(session_folder) if os.path.isfile(os.path.join(session_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
         try:
            data = TimestampedPacketSessionData_pb2.TimestampedPacketSessionData()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            session_packets.append(data)
         except:
            f.close()
            print("Could not read session packet file %s." %(filepath))
            continue
   return session_packets