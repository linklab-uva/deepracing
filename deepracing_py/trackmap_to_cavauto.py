
import deepracing, deepracing_models, deepracing_models.math_utils as mu
import deepracing.path_utils.pcd_utils as pcd_utils
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import scipy.interpolate
import matplotlib.pyplot as plt
import shutil
import torch
import yaml

def trackmap_to_cavauto(trackname : str, outdir : str, search_dirs : list[str] | None, flatten : bool, speed_factor : float):
    print("Getting trackmap %s" % (trackname,))
    if search_dirs is None:
        search_dirs = []

    all_search_dirs = search_dirs + os.environ["F1_MAP_DIRS"].split(os.pathsep)
   
    trackmap = deepracing.searchForTrackmap(trackname, all_search_dirs, align=True, transform_to_map=True)

    if trackmap is None:
        raise ValueError("Trackmap for name %s not found" % (trackname,))
    
    rl = np.concatenate([trackmap.raceline[k] for k in ["x","y","z"]], axis=1)
    ib = np.concatenate([trackmap.inner_boundary[k] for k in ["x","y","z"]], axis=1)
    ob = np.concatenate([trackmap.outer_boundary[k] for k in ["x","y","z"]], axis=1)
    cl = np.concatenate([trackmap.width_map[k] for k in ["x","y","z"]], axis=1)
    


    if flatten:
        pca : PCA = PCA(n_components=2)
        pca.fit(np.concatenate([rl,ib,ob,cl], axis=0)) 

        zvec = np.cross(pca.components_[0], pca.components_[1])
        zvec*=np.sign(zvec[-1])

        cl_rt = pca.inverse_transform(pca.transform(cl))
        xvec = cl_rt[1]-cl_rt[0]
        xvec = xvec/np.linalg.norm(xvec, ord=2)

        yvec = np.cross(zvec,xvec)
        yvec = yvec/np.linalg.norm(yvec, ord=2)  


        xvec_true = np.cross(yvec,zvec)
        xvec_true = xvec_true/np.linalg.norm(xvec_true, ord=2)

        Rflatten = Rotation.from_matrix(np.stack([xvec_true, yvec, zvec], axis=0))
        Tflatten = -Rflatten.apply(cl[0])

        rl = Rflatten.apply(rl) + Tflatten
        ib = Rflatten.apply(ib) + Tflatten
        ob = Rflatten.apply(ob) + Tflatten
        cl = Rflatten.apply(cl) + Tflatten

        rl[:,-1]=ib[:,-1]=ob[:,-1]=cl[:,-1] = 0.0
    

    rltime = trackmap.raceline["time"][:,0]
    rl_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(rltime, rl, k=3, bc_type="periodic")
    rl_vels = rl_spline(rltime, nu=1)
    rl_speeds = speed_factor*np.linalg.norm(rl_vels, ord=2, axis=1, keepdims=True)
    rl_aug = np.concatenate([rl, rl_speeds], axis=1)

    fig, ax = plt.subplots()
    ibartist, = ax.plot(*(ib[:,:2].T), color="black")
    obartist, = ax.plot(*(ob[:,:2].T), color=ibartist.get_color())
    rlartist, = ax.plot(*(rl[:,:2].T), color="green")
    clartist, = ax.plot(*(cl[:,:2].T), color="blue")
    ax.set_aspect(aspect="equal", adjustable="datalim")
    plt.show()  


    if trackmap.clockwise:
        lb=ob
        rb=ib
    else:
        lb=ib
        rb=ob

    tracknameout = "deepracing-" + trackname.lower()
    outdirnorm = os.path.join(os.path.normpath(outdir), tracknameout)
    if os.path.isdir(outdirnorm):
        shutil.rmtree(outdirnorm)
    os.makedirs(outdirnorm)
    fmt="%4.3f"
    delimiter=","
    cl_aug = np.concatenate([cl, 10.0*np.ones_like(cl[:,[0,]])], axis=1)
    roll_map = cl_aug.copy()
    roll_map[:,-1] = 0.0
    with open(os.path.join(outdirnorm, "center_line.csv"), "w") as f:
        np.savetxt(f, cl_aug, fmt=fmt, delimiter=delimiter)
    with open(os.path.join(outdirnorm, "inner_bound.csv"), "w") as f:
        np.savetxt(f, lb, fmt=fmt, delimiter=delimiter)
    with open(os.path.join(outdirnorm, "outer_bound.csv"), "w") as f:
        np.savetxt(f, rb, fmt=fmt, delimiter=delimiter)
    with open(os.path.join(outdirnorm, "raceline.csv"), "w") as f:
        np.savetxt(f, rl_aug, fmt=fmt, delimiter=delimiter)
    with open(os.path.join(outdirnorm, "roll_map.csv"), "w") as f:
        np.savetxt(f, roll_map, fmt=fmt, delimiter=delimiter)

    if not flatten:
        exit(0)

    print("Making 2d version of deepracing trackmap")
    cavsim_trackmap_name ="%s_cavsim" % (trackname,)
    dr_trackmout_outdir = os.path.join(os.path.normpath(outdir), cavsim_trackmap_name)
    if os.path.isdir(dr_trackmout_outdir):
        shutil.rmtree(dr_trackmout_outdir)
    os.makedirs(dr_trackmout_outdir)

    print("Building path helpers")
    dr_samp = 2.0

    ib = torch.as_tensor(ib[:,[0,1]]).double().cuda()
    ob = torch.as_tensor(ob[:,[0,1]]).type_as(ib)
    cl = torch.as_tensor(cl[:,[0,1]]).type_as(ib)
    rl = torch.as_tensor(rl[:,[0,1]]).type_as(ib)
    rl_speeds = torch.as_tensor(rl_speeds).type_as(ib)


    innerbound_helper = mu.SimplePathHelper.from_closed_path(ib, dr_samp)
    outerbound_helper = mu.SimplePathHelper.from_closed_path(ob, dr_samp)
    centerline_helper = mu.SimplePathHelper.from_closed_path(cl, dr_samp)

    print("Built path helpers")

    _, cl_tangents, _ = centerline_helper(centerline_helper.__arclengths_in__)
    cl_tangents = cl_tangents/torch.linalg.vector_norm(cl_tangents, ord=2, dim=-1, keepdim=True)
    cl_normals = cl_tangents[:,[1,0]].clone()
    cl_normals[:,0] *= -1.0
    cl_r = centerline_helper.__arclengths_in__.detach().clone()

    cl_rotmats = torch.stack([cl_tangents, cl_normals], dim=-1)

    cl_rotmats_full = torch.zeros([cl_rotmats.shape[0], 3, 3]).type_as(cl_rotmats)
    cl_rotmats_full[:,:2,:2] = cl_rotmats
    cl_rotmats_full[:,2,2] = 1.0

    cl_rots = Rotation.from_matrix(cl_rotmats_full.cpu().numpy())
    cl_quats : np.ndarray = cl_rots.as_quat()

    # print("cl.shape", cl.shape)
    print("cl_quats:\n", cl_quats)
    # print("cl_tangents.shape", cl_tangents.shape)
    # print("centerline_helper.__arclengths_in__.shape", centerline_helper.__arclengths_in__.shape)
    ib_intersection_r = innerbound_helper.y_axis_intersection(cl, cl_rotmats)
    ib_intersection_points, _, _ = innerbound_helper(ib_intersection_r)
    ib_distances = -torch.linalg.vector_norm(ib_intersection_points - cl, ord=2, dim=-1)

    ib_lapdistances = centerline_helper.closest_point(ib) % cl_r[-1]


    ob_intersection_r = outerbound_helper.y_axis_intersection(cl, cl_rotmats)
    ob_intersection_points, _, _ = outerbound_helper(ob_intersection_r)
    ob_distances = torch.linalg.vector_norm(ob_intersection_points - cl, ord=2, dim=-1)
    ob_lapdistances = centerline_helper.closest_point(ob) % cl_r[-1]

    rl_lapdistances = centerline_helper.closest_point(rl) % cl_r[-1]



    print(ib_distances)
    print(ob_distances)

    ib_lapdistances[0] = ob_lapdistances[0] = rl_lapdistances[0] = 0.0

    if ib_lapdistances[-1] < ib_lapdistances[-20]:
        ib_lapdistances[-1]+=cl_r[-1]
    if ob_lapdistances[-1] < ob_lapdistances[-20]:
        ob_lapdistances[-1]+=cl_r[-1]
    if rl_lapdistances[-1] < rl_lapdistances[-20]:
        rl_lapdistances[-1]+=cl_r[-1]

    print(ib_lapdistances)
    print(ob_lapdistances)
    raceline_helper = mu.RacelineHelper.from_closed_path(rl, rl_speeds.squeeze(-1), dr_samp)

    boundary_type_map = {"x" : "f4", "y" : "f4", "z" : "f4", "lapdistance" : "f4"}
    
    ib_structured = np.zeros(ib.shape[0], dtype=list(boundary_type_map.items()))
    ib_structured["x"] = ib[:,0].cpu().float().numpy()
    ib_structured["y"] = ib[:,1].cpu().float().numpy()
    ib_structured["lapdistance"] = ib_lapdistances.cpu().float().numpy()
    pcd_utils.structurednumpyToPCD(ib_structured, os.path.join(dr_trackmout_outdir, "inner_boundary.pcd"))

    ob_structured = np.zeros(ob.shape[0], dtype=list(boundary_type_map.items()))
    ob_structured["x"] = ob[:,0].cpu().float().numpy()
    ob_structured["y"] = ob[:,1].cpu().float().numpy()
    ob_structured["lapdistance"] = ob_lapdistances.cpu().float().numpy()
    pcd_utils.structurednumpyToPCD(ob_structured, os.path.join(dr_trackmout_outdir, "outer_boundary.pcd"))


    widthmap_type_map = {k: "f4" for k in ["x", "y", "z", "i", "j", "k", "w", "r", "ib_distance", "ob_distance"]}
    widthmap_structured = np.zeros(cl.shape[0], dtype=list(widthmap_type_map.items()))
    widthmap_structured["x"] = cl[:,0].cpu().float().numpy()
    widthmap_structured["y"] = cl[:,1].cpu().float().numpy()
    widthmap_structured["i"] = cl_quats[:,0].astype(np.float32)
    widthmap_structured["j"] = cl_quats[:,1].astype(np.float32)
    widthmap_structured["k"] = cl_quats[:,2].astype(np.float32)
    widthmap_structured["w"] = cl_quats[:,3].astype(np.float32)
    widthmap_structured["r"] = cl_r.cpu().float().numpy()
    widthmap_structured["ib_distance"] = ib_distances.cpu().float().numpy()
    widthmap_structured["ob_distance"] = ob_distances.cpu().float().numpy()
    pcd_utils.structurednumpyToPCD(widthmap_structured, os.path.join(dr_trackmout_outdir, "widthmap.pcd"))


    raceline_type_map = {k: "f4" for k in ["x", "y", "z", "lapdistance", "time"]}
    raceline_structured = np.zeros(rl.shape[0], dtype=list(raceline_type_map.items()))
    raceline_structured["x"] = rl[:,0].cpu().float().numpy()
    raceline_structured["y"] = rl[:,1].cpu().float().numpy()
    raceline_structured["lapdistance"] = rl_lapdistances.cpu().float().numpy()
    raceline_structured["time"] = rltime.astype(np.float32)
    pcd_utils.structurednumpyToPCD(raceline_structured, os.path.join(dr_trackmout_outdir, "raceline.pcd"))

    config_2d = {
        "clockwise": trackmap.clockwise,
        "name": cavsim_trackmap_name,
        "startingline_pose": {
            "position": [0.0, 0.0, 0.0],
            "quaternion": [0.0, 0.0, 0.0, 1.0]
        },
        "startinglinewidth": torch.linalg.vector_norm(ib[0]-ob[0], ord=2).item(),
        "tracklength": cl_r[-1].item()
    }
    with open(os.path.join(dr_trackmout_outdir, "metadata.yaml"), "w") as f:
        yaml.safe_dump(config_2d, f)
    # clockwise: true
    # name: Australia
    # startingline_pose:
    #     position:
    #     - -109.20646306844381
    #     - 2.8839973440976023
    #     - 463.5367687659389
    #     quaternion:
    #     - -0.27083737246629574
    #     - 0.6535311225845039
    #     - 0.6526880456996641
    #     - 0.27118721299084886
    # startinglinewidth: 11.854428882218064
    # tracklength: 5276.6729743973965

        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="TrackMap to CavAuto")
    parser.add_argument("trackmap", type=str)
    parser.add_argument("outdir", type=str)
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--search-dirs", type=str, nargs="+", default=None)

    args = parser.parse_args()
    argdict = vars(args)

    keys=["trackmap", "outdir", "search_dirs", "flatten", "speed_factor"]
    trackmap_to_cavauto(*[argdict[k] for k in keys])
