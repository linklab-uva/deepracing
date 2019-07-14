import TimestampedPacketMotionData_pb2
def getAllMotionPackets(motion_data_folder: str, use_json: bool):
   motion_packets = []
   if use_json:
      filepaths = [os.path.join(motion_data_folder, f) for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]
      jsonstrings = [(open(path, 'r')).read() for path in filepaths]
      for jsonstring in jsonstrings:
         data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
         google.protobuf.json_format.Parse(jsonstring, data)
         motion_packets.append(data)
   else:
      filepaths = [os.path.join(motion_data_folder, f) for f in os.listdir(motion_data_folder) if os.path.isfile(os.path.join(motion_data_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
      for filepath in filepaths:
         try:
            data = TimestampedPacketMotionData_pb2.TimestampedPacketMotionData()
            f = open(filepath,'rb')
            data.ParseFromString(f.read())
            f.close()
            motion_packets.append(data)
         except:
            print("Could not read udp file %s." %(filepath))
            continue
   return motion_packets