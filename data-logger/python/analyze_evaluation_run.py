
import TimestampedPacketMotionData_pb2, TimestampedPacketCarTelemetryData_pb2
import argparse
import argcomplete
import os
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import Polygon
from shapely.geometry import LinearRing
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
def analyzedatasets(main_dir,subdirs, prefix, results_dir="results", plot=False, json=False):
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
        lapdata_packets = sorted(proto_utils.getAllLapDataPackets(lapdata_dir,use_json=json), key=eval_utils.udpPacketKey)
        print("Got %d lap data packets" %(len(lapdata_packets),))
        final_lap_packet = lapdata_packets[-1]
        final_lap_data = final_lap_packet.udp_packet.m_lapData[0]
        if final_lap_data.m_currentLapNum>1:
          laptimes[i] = float(final_lap_data.m_lastLapTime)
        else:
          laptimes[i] = np.nan

        #print(final_lap_data)
        motion_packets, failurescores, failuretimes, failuretimediffs, failuredistances, failuredistancediffs, velocities, cummulativedistance \
            = eval_utils.evalDataset(dset_dir,\
            "../tracks/Australia_innerlimit.track", "../tracks/Australia_outerlimit.track", plot=plot, json=json)
      #  if(failurescores is None):
        fig : matplotlib.figure.Figure = plt.figure()
        try:   
          velocity_norms = 3.6*la.norm(velocities,axis=1)
          sessiontime_array = np.array([p.udp_packet.m_header.m_sessionTime for p in motion_packets])
          sessiontime_array = sessiontime_array - sessiontime_array[0]
        # axes : matplotlib.axes.Axes = fig.add_axes()
          plt.plot(cummulativedistance, velocity_norms, figure=fig)
          plt.xlabel("Distance (meters)", figure=fig)
          plt.ylabel("Velocity (kilometer/hour)", figure=fig)
          plt.title("Velocity versus Distance (Run %d)" %(i+1,), figure=fig)
          fig.savefig( os.path.join( output_dir, "velplot_distance_run_%d.png" % (i+1,) ), bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_distance_run_%d.eps" % (i+1,) ), format='eps', bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_distance_run_%d.pdf" % (i+1,) ), format='pdf', bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_distance_run_%d.svg" % (i+1,) ), format='svg', bbox_inches='tight')
          del fig
          fig : matplotlib.figure.Figure = plt.figure()
        # axes : matplotlib.axes.Axes = fig.add_axes()
          plt.plot(sessiontime_array, velocity_norms, figure=fig)
          plt.xlabel("Session Time (seconds)", figure=fig)
          plt.ylabel("Velocity (kilometer/hour)", figure=fig)
          plt.title("Velocity versus Session Time (Run %d)" %(i+1,), figure=fig)
          fig.savefig( os.path.join( output_dir, "velplot_time_run_%d.png" % (i+1,) ), bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_time_run_%d.eps" % (i+1,) ), format='eps', bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_time_run_%d.pdf" % (i+1,) ), format='pdf', bbox_inches='tight')
          fig.savefig( os.path.join( output_dir, "velplot_time_run_%d.svg" % (i+1,) ), format='svg', bbox_inches='tight')
          thiswillfailifnone = failurescores[0]
          mtbf[i] = np.mean(failuretimediffs)
          mdbf[i] = np.mean(failuredistancediffs)
          mean_failure_scores[i] = np.mean(failurescores)
          num_failures[i] = float(failuredistances.shape[0])
          del fig
        except:
          mtbf[i] = np.nan
          mdbf[i] = np.nan
          mean_failure_scores[i] = np.nan
          del fig
        # print( "Number of failures: %d" % ( num_failures[i] ) )
        # print( "Mean time between failures: %f" % ( mtbf[i] ) )
        # print( "Mean failure distance: %f" % ( mean_failure_distances[i] ) )
    resultsdict["mean_failure_scores"] = mean_failure_scores[mean_failure_scores==mean_failure_scores].tolist()
    resultsdict["num_failures"] = num_failures[num_failures==num_failures].tolist()
    resultsdict["mean_time_between_failures"] = mtbf[mtbf==mtbf].tolist()
    resultsdict["mean_distance_between_failures"] = mdbf[mdbf==mdbf].tolist()

    goodlaps = laptimes[laptimes==laptimes]
    resultsdict["laptimes"] = goodlaps.tolist()
    resultsdict["num_successful_laps"] = len(goodlaps)
    resultsdict["grandmean_laptimes"] = float(np.mean(goodlaps))
    resultsdict["grandmean_failure_scores"] = float(np.mean(np.array(resultsdict["mean_failure_scores"])))
    resultsdict["grandmean_num_failures"] = float(np.mean(np.array(resultsdict["num_failures"])))
    resultsdict["grandmean_time_between_failures"] = float(np.mean(np.array(resultsdict["mean_time_between_failures"])))
    resultsdict["grandmean_distance_between_failures"] = float(np.mean(np.array(resultsdict["mean_distance_between_failures"])))
    print("\n", flush=True)
    print("Results for %s:"%(prefix))
    print( resultsdict , flush=True)
    print("\n", flush=True)
    with open(results_fp,'w') as f:
        yaml.dump(resultsdict,f,Dumper=yaml.SafeDumper)
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
json=True
analyzedatasets(main_dir,bezier_dsets,"Bezier_Predictor",results_dir=results_dir,json=True)
analyzedatasets(main_dir,waypoint_dsets,"Waypoint_Predictor",results_dir=results_dir,json=True)
analyzedatasets(main_dir,cnnlstm_dsets,"CNNLSTM",results_dir=results_dir,json=False)
analyzedatasets(main_dir,pilotnet_dsets,"PilotNet",results_dir=results_dir,json=False)
# exit(0)
rinner, Xinner = proto_utils.loadTrackfile("../tracks/Australia_innerlimit.track")
router, Xouter = proto_utils.loadTrackfile("../tracks/Australia_outerlimit.track")
rraceline, Xraceline = proto_utils.loadTrackfile("../tracks/Australia_racingline.track")
racelinepolygon : Polygon = Polygon(Xraceline[:,[0,2]].tolist())
# plt.plot(*racelinepolygon.exterior.xy)
# plt.show()
output_dir = os.path.join(main_dir,results_dir)
for i in range(1,runmax+1):
    if i == 3:
      continue

    distances_bezier, throttles_bezier, steering_bezier, velocities_bezier, positions_bezier, rotations_bezier, positions_bezier_true, rotations_bezier_true = eval_utils.getRacePlots(os.path.join(main_dir,bezier_dsets[i-1]), json=True)
    speeds_bezier = la.norm(velocities_bezier,axis=1)
    raceline_errors_bezier = np.array([ eval_utils.polyDist(racelinepolygon, Point(positions_bezier_true[j,[0,2]].tolist())) for j in range(positions_bezier_true.shape[0]) ])
    
    distances_waypoint, throttles_waypoint, steering_waypoint, velocities_waypoint, positions_waypoint, rotations_waypoint, positions_waypoint_true, rotations_waypoint_true = eval_utils.getRacePlots(os.path.join(main_dir,waypoint_dsets[i-1]), json=True)
    speeds_waypoint = la.norm(velocities_waypoint,axis=1)
    raceline_errors_waypoint = np.array([ eval_utils.polyDist(racelinepolygon, Point(positions_waypoint_true[j,[0,2]].tolist())) for j in range(positions_waypoint_true.shape[0]) ])

    distances_pilotnet, throttles_pilotnet, steering_pilotnet, velocities_pilotnet, positions_pilotnet, rotations_pilotnet, positions_pilotnet_true, rotations_pilotnet_true = eval_utils.getRacePlots(os.path.join(main_dir,pilotnet_dsets[i-1]), json=False)
    speeds_pilotnet = la.norm(velocities_pilotnet,axis=1)
    raceline_errors_pilotnet = np.array([ eval_utils.polyDist(racelinepolygon, Point(positions_pilotnet_true[j,[0,2]].tolist())) for j in range(positions_pilotnet_true.shape[0]) ])

    distances_cnnlstm, throttles_cnnlstm, steering_cnnlstm, velocities_cnnlstm, positions_cnnlstm, rotations_cnnlstm, positions_cnnlstm_true, rotations_cnnlstm_true = eval_utils.getRacePlots(os.path.join(main_dir,cnnlstm_dsets[i-1]), json=False)
    speeds_cnnlstm = la.norm(velocities_cnnlstm,axis=1)
    raceline_errors_cnnlstm = np.array([ eval_utils.polyDist(racelinepolygon, Point(positions_cnnlstm_true[j,[0,2]].tolist())) for j in range(positions_cnnlstm_true.shape[0]) ])

    xmin = 590
    xmax = 900
    zmin = 570
    zmax = 870
    Xinnerisolated = np.array([Xinner[i,:] for i in range(Xinner.shape[0])\
     if Xinner[i,0]>xmin and Xinner[i,0]<xmax and Xinner[i,2]>zmin and Xinner[i,2]<zmax ])
    Xouterisolated = np.array([Xouter[i,:] for i in range(Xouter.shape[0])\
     if Xouter[i,0]>xmin and Xouter[i,0]<xmax and Xouter[i,2]>zmin and Xouter[i,2]<zmax ])
    positions_bezier_isolated = np.array([positions_bezier[i,:] for i in range(positions_bezier.shape[0])\
     if positions_bezier[i,0]>xmin and positions_bezier[i,0]<xmax and positions_bezier[i,2]>zmin and positions_bezier[i,2]<zmax ])
    positions_waypoint_isolated = np.array([positions_waypoint[i,:] for i in range(positions_waypoint.shape[0])\
     if positions_waypoint[i,0]>xmin and positions_waypoint[i,0]<xmax and positions_waypoint[i,2]>zmin and positions_waypoint[i,2]<zmax ])

    figtroublearea : matplotlib.figure.Figure = plt.figure(frameon=True)
    plt.plot(Xinnerisolated[:,0], Xinnerisolated[:,2], color='gray', label='Inner Boundary')
    plt.plot(Xouterisolated[:,0], Xouterisolated[:,2], color='black', label='Outer Boundary')
    plt.plot(positions_bezier_isolated[:,0], positions_bezier_isolated[:,2], color='blue', label='Bezier Predictor Path')
    #plt.legend(fontsize='large')

    # figbezier.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_bezier_run_%d.png" % (i,) ), bbox_inches='tight')
    # figbezier.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_bezier_run_%d.eps" % (i,) ), format='eps', bbox_inches='tight')
    # figbezier.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_bezier_run_%d.pdf" % (i,) ), format='pdf', bbox_inches='tight')
    # figbezier.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_bezier_run_%d.svg" % (i,) ), format='svg', bbox_inches='tight')  
    # del figbezier
    # figwaypoint : matplotlib.figure.Figure = plt.figure(frameon=True)


   # plt.plot(Xinnerisolated[:,0], Xinnerisolated[:,2], color='gray', label='Inner Boundary')
    #plt.plot(Xouterisolated[:,0], Xouterisolated[:,2], color='black', label='Outer Boundary')
    plt.plot(positions_waypoint_isolated[:,0], positions_waypoint_isolated[:,2], color='red', label='Waypoint Predictor Path')
    plt.legend(fontsize='large')
    print(positions_bezier.shape)
    print(positions_waypoint.shape)
    os.makedirs(os.path.join( output_dir, "trouble_area_plots"), exist_ok=True)
    figtroublearea.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_run_%d.png" % (i,) ), bbox_inches='tight')
    figtroublearea.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_run_%d.eps" % (i,) ), format='eps', bbox_inches='tight')
    figtroublearea.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_run_%d.pdf" % (i,) ), format='pdf', bbox_inches='tight')
    figtroublearea.savefig( os.path.join( output_dir, "trouble_area_plots", "trouble_area_run_%d.svg" % (i,) ), format='svg', bbox_inches='tight')
    del figtroublearea


    figcombined : matplotlib.figure.Figure = plt.figure(frameon=True)
    plt.scatter(positions_pilotnet[:,2], positions_pilotnet[:,0], color='teal', label='PilotNet', marker='+')
    plt.scatter(positions_cnnlstm[:,2], positions_cnnlstm[:,0], color='gray', label='CNNLSTM', marker='o')
    plt.plot(positions_bezier[:,2], positions_bezier[:,0], color='blue', label='Bezier Predictor (does not crash)')
    plt.scatter(positions_pilotnet[-1,2], positions_pilotnet[-1,0], color='red', label='PilotNet Crash Point', marker='X')
    plt.scatter(positions_cnnlstm[-1,2], positions_cnnlstm[-1,0], color='black', label='CNNLSTM Crash Point', marker='X')
    plt.legend(fontsize='large')
    os.makedirs(os.path.join( output_dir, "total_path_comparison_plots"), exist_ok=True)
    figcombined.savefig( os.path.join( output_dir, "total_path_comparison_plots", "total_path_comparison_run_%d.png" % (i,) ), bbox_inches='tight')
    figcombined.savefig( os.path.join( output_dir, "total_path_comparison_plots", "total_path_comparison_run_%d.eps" % (i,) ), format='eps', bbox_inches='tight')
    figcombined.savefig( os.path.join( output_dir, "total_path_comparison_plots", "total_path_comparison_run_%d.pdf" % (i,) ), format='pdf', bbox_inches='tight')
    figcombined.savefig( os.path.join( output_dir, "total_path_comparison_plots", "total_path_comparison_run_%d.svg" % (i,) ), format='svg', bbox_inches='tight')
    del figcombined
    # print(len(motion_packets_bezier))
    # print(len(lap_packets_bezier))
    # print(len(telemetry_packets_bezier))
    figraceline_errors : matplotlib.figure.Figure = plt.figure(frameon=True)
    os.makedirs(os.path.join( output_dir, "raceline_errors"), exist_ok=True)
    plt.plot(distances_bezier, raceline_errors_bezier, color='blue', label='Bezier Predictor')
    plt.plot(distances_waypoint, raceline_errors_waypoint, color='red', label='Waypoint Predictor')
    plt.legend(fontsize='large')
    figraceline_errors.savefig( os.path.join( output_dir, "raceline_errors", "raceline_errors_run_%d.png" % (i,) ), bbox_inches='tight')
    figraceline_errors.savefig( os.path.join( output_dir, "raceline_errors", "raceline_errors_run_%d.eps" % (i,) ), format='eps', bbox_inches='tight')
    figraceline_errors.savefig( os.path.join( output_dir, "raceline_errors", "raceline_errors_run_%d.pdf" % (i,) ), format='pdf', bbox_inches='tight')
    figraceline_errors.savefig( os.path.join( output_dir, "raceline_errors", "raceline_errors_run_%d.svg" % (i,) ), format='svg', bbox_inches='tight')
    del figraceline_errors


    idxskip=100
    fig : matplotlib.figure.Figure = plt.figure(frameon=True)
    ax1 : matplotlib.axes.Axes = fig.add_subplot(211)
    ax3 : matplotlib.axes.Axes = fig.add_subplot(212)
    fig.suptitle('Speed And Steering versus Distance')

    beziercolor='teal'
    waypointcolor='darkslategray'
    bezierlabel = 'Bezier Curve Predictor'
    waypointlabel = 'Waypoint Predictor'
    #legendpos = 'lower left'
    legendpos = (0.4,0.65)
    linesbezier = ax1.plot(distances_bezier[idxskip:] - distances_bezier[idxskip], speeds_bezier[idxskip:],c=beziercolor,label=bezierlabel)
    lineswaypoint = ax1.plot(distances_waypoint[idxskip:] - distances_waypoint[idxskip], speeds_waypoint[idxskip:],c=waypointcolor,label=waypointlabel)
    #ax1.legend(loc=legendpos)
    

    
    ax3.plot(distances_bezier[idxskip:] - distances_bezier[idxskip], steering_bezier[idxskip:],c=beziercolor,label=bezierlabel)
    ax3.plot(distances_waypoint[idxskip:] - distances_waypoint[idxskip], steering_waypoint[idxskip:],c=waypointcolor,label=waypointlabel)
    ax3.legend(loc=legendpos)
  #  fig.legend(loc=legendpos, handles=lineswaypoint )
    #fig.legend(loc=legendpos, handles=linesbezier )
    

    ax1.set_ylabel("Speed (kph)")

    secax3 = ax3.secondary_yaxis('right')
    secax3.set_ylabel("Steering ([-1,1])")
    ax3.set_xlabel("Distance (meters)")

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xaxis(), visible=False)
    plt.setp(ax3.get_yaxis(), visible=False)

    fig.set_frameon(True)
    fig.subplots_adjust(hspace=0.0)

    os.makedirs(os.path.join( output_dir, "f1_style_comparisons"), exist_ok=True)
    fig.savefig( os.path.join( output_dir, "f1_style_comparisons", "comparison_distance_run_%d.png" % (i,) ), bbox_inches='tight')
    fig.savefig( os.path.join( output_dir, "f1_style_comparisons", "comparison_distance_run_%d.eps" % (i,) ), format='eps', bbox_inches='tight')
    fig.savefig( os.path.join( output_dir, "f1_style_comparisons", "comparison_distance_run_%d.pdf" % (i,) ), format='pdf', bbox_inches='tight')
    fig.savefig( os.path.join( output_dir, "f1_style_comparisons", "comparison_distance_run_%d.svg" % (i,) ), format='svg', bbox_inches='tight')
    del fig
    
    # print(len(motion_packets_waypoint))
    # print(len(lap_packets_waypoint))
    # print(len(telemetry_packets_waypoint))

