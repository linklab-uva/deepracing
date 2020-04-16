import argparse
import numpy as np
import scipy, scipy.stats
import deepracing, deepracing_models
import google.protobuf.json_format as json_utils
import yaml
import os
import typing
from typing import List
from TimestampedPacketCarTelemetryData_pb2 import TimestampedPacketCarTelemetryData
from PacketCarTelemetryData_pb2 import PacketCarTelemetryData
from CarTelemetryData_pb2 import CarTelemetryData
from deepracing.protobuf_utils import getAllTelemetryPackets
import matplotlib.pyplot as plt
def packetsortkey(packet):
    return packet.timestamp
def go(args):
    argsdict = dict(args.__dict__)
    print(argsdict)
    dataset_root = argsdict["dataset_root"]
    output_dir = argsdict["output_dir"]
    car_index = argsdict["car_index"]
    tmax = argsdict["tmax"]
    print(dataset_root)
    config_file_path = os.path.join(dataset_root,"config.yaml")
    with open(config_file_path,"r") as f:
        config = yaml.load(f,Loader=yaml.SafeLoader)
    print(config)
    telemetry_folder = os.path.join(dataset_root,"udp_data","car_telemetry_packets")
    telemetry_packets : List[TimestampedPacketCarTelemetryData] = getAllTelemetryPackets(telemetry_folder, True)
    telemetry_packets : List[TimestampedPacketCarTelemetryData] = sorted(telemetry_packets, key=packetsortkey)
    timestamps : np.ndarray  = np.array([packetsortkey(telemetry_packet) for telemetry_packet in telemetry_packets])
    steering_angles : np.ndarray = -np.array([telemetry_packet.udp_packet.m_carTelemetryData[car_index].m_steer/100.0 for telemetry_packet in telemetry_packets])

    

    
    time_start = config["time_start"]
    control_delta = config["control_delta"]
    timediff = config["timediff"]



    I = timestamps>=time_start#+1*timediff
    timestamps_clipped = timestamps[I]
    steering_angles_clipped = steering_angles[I]

    
    I2 = timestamps_clipped<=tmax
    timestamps_clipped = timestamps_clipped[I2]
    steering_angles_clipped = steering_angles_clipped[I2]

    slope_ideal = control_delta/timediff
    xintercept_ideal = time_start
    yintercept_ideal = -slope_ideal*xintercept_ideal

    slope, yintercept, r_value, p_value, std_err = scipy.stats.linregress(timestamps_clipped,steering_angles_clipped)
    xintercept = -yintercept/slope
    print("Slope: %f" %(slope,))
    print("Y-Intercept: %f" %(yintercept,))
    print("X-Intercept: %f" %(xintercept,))
    print("Measured tstart: %f" %(time_start,))
    print("Expected Slope: %f" %(-control_delta/timediff,))
    print("Actuals x intercept differs from expected by : %f milliseconds" %(time_start - xintercept,))



   # plt.plot(timestamps,steering_angles)
    dt = 100
    tplot = np.linspace(time_start-dt,timestamps_clipped[-1]+dt,1000)
    plt.scatter(timestamps_clipped,steering_angles_clipped, facecolors='none', edgecolors="black", marker='o', label = "Measured Data")
    #plt.plot(tplot,tplot*slope_ideal + yintercept_ideal, color="blue", label="Ideal Line")
    plt.plot(tplot,tplot*slope + yintercept, color="red", label="Regression Line")
    plt.axvline(x=time_start,ymin=0.0,ymax=1.0, color="green", label="Expected Start Time")
    plt.axhline(y=0.0, color="black", label="steering angle=0")
    plt.legend(loc = (0.15,0.65))
    #plt.legend(loc = 'upper left')
    plt.xlabel("System Time (milliseconds)")
    plt.ylabel("Normalized Steering Angle ([-1,1])")
    plt.title("Regression Line & Expected X-intercept")
    plt.savefig(os.path.join(output_dir,"lagtest_regression.eps"))
    plt.savefig(os.path.join(output_dir,"lagtest_regression.png"))
    plt.savefig(os.path.join(output_dir,"lagtest_regression.pdf"))
    plt.savefig(os.path.join(output_dir,"lagtest_regression.svg"))
    plt.show()

if __name__=="__main__":
    parser : argparse.ArgumentParser = argparse.ArgumentParser("Test a run of the vjoy stuff")
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--tmax", type=float, default=1E9)
    parser.add_argument("--car_index", type=int, default=0)

    args = parser.parse_args()
    go(args)
    