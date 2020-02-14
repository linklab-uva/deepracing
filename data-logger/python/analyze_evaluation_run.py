
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import argcomplete
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import deepracing.protobuf_utils as proto_utils
import scipy
import scipy.spatial
import deepracing.evaluation_utils as eval_utils

import yaml
from matplotlib import pyplot as plt
import numpy.linalg as la
import matplotlib.figure
import matplotlib.axes
def analyzedatasets(main_dir,subdirs, prefix, results_dir="results", plot=False):
    numruns = len(subdirs)
    mtbf = np.zeros(numruns)
    mdbf = np.zeros(numruns)
    laptimes = np.zeros(numruns)
    mean_failure_scores = np.zeros(numruns)
    num_failures = np.zeros(numruns)
    output_dir = os.path.join(main_dir, results_dir, prefix)
    os.makedirs(output_dir,exist_ok=True)
    results_fp = os.path.join(output_dir, "results.yaml")
    resultsdict : dict = {}
    for (i, dset) in enumerate(subdirs):
        print("Running dataset %d for %s:"%(i+1, prefix), flush=True)
        dset_dir = os.path.join(main_dir, dset)
        lapdata_dir = os.path.join(dset_dir, "udp_data", "lap_packets")
        lapdata_packets = sorted(proto_utils.getAllLapDataPackets(lapdata_dir), key=eval_utils.udpPacketKey)
        print("Got %d lap data packets" %(len(lapdata_packets),))
        final_lap_packet = lapdata_packets[-1]
        final_lap_data = final_lap_packet.udp_packet.m_lapData[0]
        laptimes[i] = float(final_lap_data.m_currentLapTime)
        #print(final_lap_data)
        motion_packets, failurescores, failuretimes, failuretimediffs, failuredistances, failuredistancediffs, velocities, cummulativedistance \
            = eval_utils.evalDataset(dset_dir,\
            "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot)
      #  if(failurescores is None):
            
        velocity_norms = 3.6*la.norm(velocities,axis=1)
        mtbf[i] = np.mean(failuretimediffs)
        mdbf[i] = np.mean(failuredistancediffs)
        mean_failure_scores[i] = np.mean(failurescores)
        num_failures[i] = float(failuredistances.shape[0])
        sessiontime_array = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets])
        sessiontime_array = sessiontime_array - sessiontime_array[0]
        fig : matplotlib.figure.Figure = plt.figure()
       # axes : matplotlib.axes.Axes = fig.add_axes()
        plt.plot(cummulativedistance, velocity_norms, figure=fig)
        plt.xlabel("Distance (meters)", figure=fig)
        plt.ylabel("Velocity (kilometer/hour)", figure=fig)
        plt.title("Velocity versus Distance (Run %d)" %(i+1,), figure=fig)
        fig.savefig( os.path.join( output_dir, "velplot_distance_run_%d.png" % (i+1,) ), bbox_inches='tight')
        del fig
        fig : matplotlib.figure.Figure = plt.figure()
       # axes : matplotlib.axes.Axes = fig.add_axes()
        plt.plot(sessiontime_array, velocity_norms, figure=fig)
        plt.xlabel("Session Time (seconds)", figure=fig)
        plt.ylabel("Velocity (kilometer/hour)", figure=fig)
        plt.title("Velocity versus Session Time (Run %d)" %(i+1,), figure=fig)
        fig.savefig( os.path.join( output_dir, "velplot_time_run_%d.png" % (i+1,) ), bbox_inches='tight')
        del fig
        # print( "Number of failures: %d" % ( num_failures[i] ) )
        # print( "Mean time between failures: %f" % ( mtbf[i] ) )
        # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
    resultsdict["mean_failure_scores"] = mean_failure_scores.tolist()
    resultsdict["num_failures"] = num_failures.tolist()
    resultsdict["mean_time_between_failures"] = mtbf.tolist()
    resultsdict["mean_distance_between_failures"] = mdbf.tolist()
    resultsdict["laptimes"] = laptimes.tolist()

    resultsdict["grandmean_laptimes"] = float(np.mean(laptimes))
    resultsdict["grandmean_failure_scores"] = float(np.mean(mean_failure_scores))
    resultsdict["grandmean_num_failures"] = float(np.mean(num_failures))
    resultsdict["grandmean_time_between_failures"] = float(np.mean(mtbf))
    resultsdict["grandmean_distance_between_failures"] = float(np.mean(mdbf))
    print(resultsdict)
    with open(results_fp,'w') as f:
        yaml.dump(resultsdict,f,Dumper=yaml.SafeDumper)

    print("\n", flush=True)
    print("Results for %s:"%(prefix))
    print( "Average Number of failures: %d" % ( np.mean(num_failures) ) , flush=True)
    print( "Overall Mean time between failures: %f" % ( np.mean(mtbf) ) , flush=True)
    print( "Overall Mean distance between failures: %f" % ( np.mean(mdbf) ) , flush=True)
    print( "Overall Mean failure score: %f" % (  np.mean(mean_failure_scores)  ) , flush=True)
    print("\n", flush=True)
    return output_dir
parser = argparse.ArgumentParser()
parser.add_argument("main_dir", help="Directory of the evaluation datasets",  type=str)
parser.add_argument("--runmax", type=int, default=5, help="How many runs to parse on each model")
args = parser.parse_args()
main_dir = args.main_dir
runmax = args.runmax
bezier_dsets = ["bezier_predictor_run%d" % i for i in range(1,runmax+1)]
waypoint_dsets = ["waypoint_predictor_run%d" % i for i in range(1,runmax+1)]
cnnlstm_dsets = ["cnnlstm_run%d" % i for i in range(1,runmax+1)]
pilotnet_dsets = ["pilotnet_run%d" % i for i in range(1,runmax+1)]
print(bezier_dsets)
print(waypoint_dsets)
print(cnnlstm_dsets)
print(pilotnet_dsets)



results_dir="results"
# analyzedatasets(main_dir,bezier_dsets,"Bezier_Predictor",results_dir=results_dir)
# analyzedatasets(main_dir,waypoint_dsets,"Waypoint_Predictor",results_dir=results_dir)
# analyzedatasets(main_dir,cnnlstm_dsets,"CNNLSTM",results_dir=results_dir)
# analyzedatasets(main_dir,pilotnet_dsets,"PilotNet",results_dir=results_dir)


output_dir = os.path.join(main_dir,results_dir)
for i in range(1,runmax+1):
    motion_dir_bezier = os.path.join(main_dir,"bezier_predictor_run%d" % (i,) ,"udp_data","motion_packets")
    lap_dir_bezier = os.path.join(main_dir,"bezier_predictor_run%d" % (i,) ,"udp_data","lap_packets")
    telemetry_dir_bezier = os.path.join(main_dir,"bezier_predictor_run%d" % (i,) ,"udp_data","car_telemetry_packets")
    motion_packets_bezier = sorted(proto_utils.getAllMotionPackets(motion_dir_bezier, False), key = eval_utils.udpPacketKey)
    lap_packets_bezier = sorted(proto_utils.getAllLapDataPackets(lap_dir_bezier, False), key = eval_utils.udpPacketKey)
    telemetry_packets_bezier = sorted(proto_utils.getAllTelemetryPackets(telemetry_dir_bezier, False), key = eval_utils.udpPacketKey)

    motion_dir_waypoint = os.path.join(main_dir, "waypoint_predictor_run%d" % (i,), "udp_data", "motion_packets")
    lap_dir_waypoint = os.path.join(main_dir, "waypoint_predictor_run%d" % (i,), "udp_data", "lap_packets")
    telemetry_dir_waypoint = os.path.join(main_dir,"waypoint_predictor_run%d" % (i,) ,"udp_data","car_telemetry_packets")
    motion_packets_waypoint = sorted(proto_utils.getAllMotionPackets(motion_dir_waypoint, False), key = eval_utils.udpPacketKey)
    lap_packets_waypoint = sorted(proto_utils.getAllLapDataPackets(lap_dir_waypoint, False), key = eval_utils.udpPacketKey)
    telemetry_packets_waypoint = sorted(proto_utils.getAllTelemetryPackets(telemetry_dir_waypoint, False), key = eval_utils.udpPacketKey)

    minIbezier = int(np.min([len(motion_packets_bezier), len(lap_packets_bezier), len(telemetry_packets_bezier)]))
    motion_packets_bezier = motion_packets_bezier[0:minIbezier]
    lap_packets_bezier = lap_packets_bezier[0:minIbezier]
    telemetry_packets_bezier = telemetry_packets_bezier[0:minIbezier]
    session_times_bezier = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets_bezier])
    session_times_bezier = session_times_bezier - session_times_bezier[0]
    laptimes_bezier =  np.array([p.udp_packet.m_lapData[0].m_currentLapTime for p in lap_packets_bezier])
    distances_bezier = np.array([p.udp_packet.m_lapData[0].m_lapDistance for p in lap_packets_bezier])
    throttles_bezier = np.array([p.udp_packet.m_carTelemetryData[0].m_throttle for p in telemetry_packets_bezier])/100.0
    steering_bezier = -np.array([p.udp_packet.m_carTelemetryData[0].m_steer for p in telemetry_packets_bezier])/100.0
    velocities_bezier = 3.6*np.array([proto_utils.extractVelocity(p.udp_packet) for p in motion_packets_bezier])
    speeds_bezier = la.norm(velocities_bezier,axis=1)
    # print(len(motion_packets_bezier))
    # print(len(lap_packets_bezier))
    # print(len(telemetry_packets_bezier))

    minIwaypoint = int(np.min([len(motion_packets_waypoint), len(lap_packets_waypoint), len(telemetry_packets_waypoint)]))
    motion_packets_waypoint = motion_packets_waypoint[0:minIwaypoint]
    lap_packets_waypoint = lap_packets_waypoint[0:minIwaypoint]
    telemetry_packets_waypoint = telemetry_packets_waypoint[0:minIwaypoint]
    session_times_waypoint = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets_waypoint])
    session_times_waypoint = session_times_waypoint - session_times_waypoint[0]
    laptimes_waypoint =  np.array([p.udp_packet.m_lapData[0].m_currentLapTime for p in lap_packets_waypoint])
    distances_waypoint = np.array([p.udp_packet.m_lapData[0].m_lapDistance for p in lap_packets_waypoint])
    throttles_waypoint = np.array([p.udp_packet.m_carTelemetryData[0].m_throttle for p in telemetry_packets_waypoint])/100.0
    steering_waypoint = -np.array([p.udp_packet.m_carTelemetryData[0].m_steer for p in telemetry_packets_waypoint])/100.0
    velocities_waypoint = 3.6*np.array([proto_utils.extractVelocity(p.udp_packet) for p in motion_packets_waypoint])
    speeds_waypoint = la.norm(velocities_waypoint,axis=1)

    idxskip=100
    fig : matplotlib.figure.Figure = plt.figure(frameon=True)
    ax1 : matplotlib.axes.Axes = fig.add_subplot(311)
    ax2 : matplotlib.axes.Axes = fig.add_subplot(312)
    ax3 : matplotlib.axes.Axes = fig.add_subplot(313)
    fig.suptitle('Throttle And Speed versus Distance')

    beziercolor='teal'
    waypointcolor='darkslategray'
    bezierlabel = 'Bezier Curve Predictor'
    waypointlabel = 'Waypoint Predictor'
    #legendpos = 'lower left'
    legendpos = (0.4,0.65)
    linesbezier = ax1.plot(distances_bezier[idxskip:] - distances_bezier[idxskip], speeds_bezier[idxskip:],c=beziercolor,label=bezierlabel)
    lineswaypoint = ax1.plot(distances_waypoint[idxskip:] - distances_waypoint[idxskip], speeds_waypoint[idxskip:],c=waypointcolor,label=waypointlabel)
    #ax1.legend(loc=legendpos)
    
    ax2.plot(distances_bezier[idxskip:] - distances_bezier[idxskip], throttles_bezier[idxskip:],c=beziercolor,label=bezierlabel)
    ax2.plot(distances_waypoint[idxskip:] - distances_waypoint[idxskip], throttles_waypoint[idxskip:],c=waypointcolor,label=waypointlabel)
    #ax2.legend(loc=legendpos)
    
    ax3.plot(distances_bezier[idxskip:] - distances_bezier[idxskip], steering_bezier[idxskip:],c=beziercolor,label=bezierlabel)
    ax3.plot(distances_waypoint[idxskip:] - distances_waypoint[idxskip], steering_waypoint[idxskip:],c=waypointcolor,label=waypointlabel)
    ax3.legend(loc=legendpos)
  #  fig.legend(loc=legendpos, handles=lineswaypoint )
    #fig.legend(loc=legendpos, handles=linesbezier )
    

    ax1.set_ylabel("Speed (meters/second)")

    #ax2.get_
    secax2 = ax2.secondary_yaxis('right')
    secax3 = ax3.secondary_yaxis('right')
    secax2.set_ylabel("Throttle ([0,1])")
    secax3.set_ylabel("Steering ([-1,1])")
    ax3.set_xlabel("Distance (meters)")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xaxis(), visible=False)
    plt.setp(ax2.get_yaxis(), visible=False)
    plt.setp(ax3.get_yaxis(), visible=False)
  #  plt.setp(fig.patch, visible=False)
  #  ax1.set_frame_on(False)
   # ax2.set_frame_on(False)
    fig.set_frameon(True)
    fig.subplots_adjust(hspace=0.0)
  #  plt.subplots_adjust(bottom=0.01, right=0.8, top=0.02)
   # fig.tight_layout()
  #  fig.set_frame_on(True)
   # plt.show()
    fig.savefig( os.path.join( output_dir, "comparison_distance_run_%d.png" % (i,) ), bbox_inches='tight')
    del fig
    
    # print(len(motion_packets_waypoint))
    # print(len(lap_packets_waypoint))
    # print(len(telemetry_packets_waypoint))

