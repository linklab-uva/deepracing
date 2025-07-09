
import os, sys
thisdir=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(thisdir, "..")))
import deepracing_models.math_utils as mu, deepracing_models.math_utils.bezier as bezier
import deepracing_models.data_loading.file_datasets.OvertakingTrajectoriesDataset as otd
import deepracing_models.math_utils.rotations as drrot
import numpy as np
import yaml
import torch, torch.distributions, torch.nn
import torch.utils.data as torchdata
import tqdm
import matplotlib.pyplot as plt, matplotlib.figure, matplotlib.axes, matplotlib.patches
from matplotlib.backends.backend_pdf import PdfPages
import time
plt.rc('font', family='serif')
plt.rc("svg", fonttype='none')
plt.rc('text.latex', preamble=
       r'\usepackage{amsmath}'\
       + "\n" + r'\usepackage{amssymb}'
       )
plt.rc('text', usetex=False)
import geoopt, geoopt.optim
from scipy.spatial.transform import Rotation
from deepracing_models.math_utils.integrate import GaussLegendre1D, GaussianIntegral2D, GaussianIntegralCirc
import deepracing_models.math_utils.collision_checking as cc
from deepracing_models.probabilistic_models import ProbabilisticBezierCurve
import scipy.stats
from matplotlib.colors import TABLEAU_COLORS, same_color
import shutil

class PBCurve(torch.nn.Module):
    def __init__(self, means : torch.Tensor):
        super(PBCurve, self).__init__()
        self.mean = torch.nn.Parameter(means, requires_grad=True)
        self.principle_vars = torch.nn.Parameter(torch.ones_like(means), requires_grad=True)
        self.sphere_manifold = geoopt.Sphere()
        self.directions = torch.nn.ParameterList([
            geoopt.ManifoldParameter(torch.as_tensor([1.0, 0.0]).type_as(means), manifold=self.sphere_manifold, requires_grad=True)
        for j in range(means.shape[0])])
        self.flip=torch.nn.Parameter(torch.as_tensor([-1.0, 1.0]).type_as(means), requires_grad=False)
        self.relu = torch.nn.ReLU()
        
      
    def covars_dense(self):

        rotmats = torch.stack(
            [ torch.stack([self.directions[j], (self.directions[j][[1,0]])*self.flip], dim=-1) for j in range(len(self.directions))],
        dim=0)
        principle_var_mats = torch.diag_embed(self.principle_vars).abs()
        covars = rotmats@principle_var_mats@rotmats.transpose(-2,-1)
        return covars
    def forward(self, M : torch.Tensor):
        # M = mu.bezierM(s, self.mean.shape[0]-1)
        peval = M@self.mean[None]
        covars = self.covars_dense()
        msquare = M.square()
        # covars_exp = covars[None].view(s.shape[0], covars.shape[0], -1)
        pointscovarout : torch.Tensor = torch.sum(msquare[...,None,None]*covars[None,None], dim=-3)
        return peval, pointscovarout
class MaxCircleProb(torch.nn.Module):
    def __init__(self, curve_covars : torch.Tensor, car_width : float, car_length : float):
        super(MaxCircleProb, self).__init__()
        car_diameter = float(np.sqrt(car_width**2 + car_length**2))
        self.car_diameter = torch.nn.Parameter(torch.as_tensor(car_diameter).double(), requires_grad=False)
        self.curve_covars = torch.nn.Parameter(curve_covars.cpu().double(), requires_grad=False)
        self.s_eval = torch.nn.Parameter(torch.linspace(0.0, 1.0, steps=2**7).double(), requires_grad=False)
        kbezier = curve_covars.shape[0]-1
        M_eval = bezier.bezierM(self.s_eval[None].detach(), kbezier).type_as(curve_covars)
        covars_eval = torch.zeros([self.s_eval.shape[0], 2, 2]).type_as(curve_covars)
        for i in range(covars_eval.shape[0]):
            for j in range(kbezier+1):
                covars_eval[i]+=(M_eval[0,i,j]**2)*(curve_covars[j])
        # self.covars_eval = torch.nn.Parameter(covars_eval.cpu().double(), requires_grad=False)

        scale_tril : torch.Tensor = torch.linalg.cholesky(covars_eval, upper=False)
        self.scale_tril = torch.nn.Parameter(scale_tril.cpu().double(), requires_grad=False)


    def forward(self, attacker_curve : torch.Tensor, defender_curve : torch.Tensor):
        kbezier = self.curve_covars.shape[0]-1
        M_eval = bezier.bezierM(self.s_eval[None], kbezier)
        attacker_eval= (M_eval@attacker_curve)[0]
        defender_eval= (M_eval@defender_curve)[0]

        # sampled_curves = probabilistic_curve.sample([int(round(1.0*(2**10))),])
        # sample_points_eval = (M_eval@sampled_curves)
        # noise_eval = torch.randn([int(round(1.0*(2**10))),*defender_eval.shape]).type_as(defender_eval)
        # noise_eval *= torch.sqrt(self.covars_eval[:,0,0])[None,:,None]
        dist_eval : torch.distributions.MultivariateNormal = torch.distributions.MultivariateNormal(defender_eval, scale_tril=self.scale_tril, validate_args=False)
        sample_points_eval = dist_eval.sample([int(round(1.0*(2**10))),])
        sample_points_delta = sample_points_eval - attacker_eval
        sample_points_deltanorms = torch.linalg.vector_norm(sample_points_delta, dim=-1)
        circleprobs = torch.sum(sample_points_deltanorms<self.car_diameter, dim=0)/sample_points_delta.shape[0]
        return circleprobs.max().clip(min=0.0, max=1.0)
        
        

def main(*args, datadir : str, debug : bool, save : str | None, out : str | None, sigma0 : float, sigma1 : float, 
         gamma : float, alpha : float, kbezier: int, car_length : float, car_width : float, n1 : int, n2 : int):
    dsets = []
    for subdir in os.listdir(datadir):
        potentialdatadir = os.path.join(datadir,subdir)
        if not os.path.isdir(potentialdatadir):
            continue
        if not os.path.isfile(os.path.join(potentialdatadir, "metadata.yaml")):
            continue
        if not os.path.isfile(os.path.join(potentialdatadir, "data.npz")):
            continue
        dsets.append(otd.from_npz_dir(potentialdatadir, kbezier))

    s = torch.linspace(0.0, 1.0, steps=60, dtype=torch.float64).cuda(0)
    desired_covars = torch.eye(2)[None].repeat(s.shape[0],1,1).type_as(s)
    desired_covars[:,0,0] = desired_covars[:,1,1] = torch.linspace(sigma0, sigma1, steps=s.shape[0]).square().type_as(s)
    mvn0 = torch.distributions.MultivariateNormal(torch.zeros_like(desired_covars[...,0]), covariance_matrix=desired_covars)

    pbcurve = ProbabilisticBezierCurve(torch.zeros(kbezier+1,2), 1E-1 + torch.eye(desired_covars.shape[-1])[None].expand(kbezier+1, 2, 2).clone()).to(dtype=s.dtype, device=s.device)
    pbcurve.mean = pbcurve.mean.requires_grad_(requires_grad=False)

    M = mu.bezierM(s[None], kbezier)
    _, covars = pbcurve(M)
    rsgd : geoopt.optim.RiemannianSGD = geoopt.optim.RiemannianSGD(pbcurve.parameters(), lr=5E-2)
    t = tqdm.tqdm(range(200)) 
    for i in t:
        rsgd.zero_grad()
        zeromeans, covars = pbcurve(M)
        covars : torch.Tensor = covars[0]
        mvn1 = torch.distributions.MultivariateNormal(zeromeans[0], covariance_matrix=covars)
        loss = torch.distributions.kl_divergence(mvn0, mvn1).sum()
        loss.backward()
        rsgd.step()
        t.set_postfix({"Loss": loss.item()})
    pbcurve.requires_grad_(False)
    if debug:
        # print("pbcurve.covars_dense().sqrt():", pbcurve.covars_dense().sqrt())
        # print("pbcurve.directions:", torch.stack([d.detach() for d in pbcurve.directions], dim=0))
        eigvals, eigvecs = torch.linalg.eigh(torch.stack([c.detach() for c in pbcurve.covariance], dim=0))
        print("eigvals.sqrt():", eigvals.sqrt())
        print("eigvecs:", eigvecs)
        # print(loss.item())
    
    curve_covars = torch.stack([pbcurve.covariance[i] for i in range(kbezier+1)]).detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).type_as(s)
    # curve_covars = pbcurve.covars_dense().detach().type_as(s)

    car_diameter = np.sqrt(car_length*car_length + car_width*car_width)
    boxpoints01 = (torch.stack(torch.meshgrid([
        0.5*torch.linspace(-car_length, car_length, steps=2),
        0.5*torch.linspace(-car_width, car_width, steps=2)
    ], indexing="ij"), dim=0)).reshape(2,-1).T.type_as(s)
    boxpoints01 = boxpoints01[torch.argsort(torch.atan2(boxpoints01[:,1], boxpoints01[:,0]))]
    for dset in dsets:
        dset.compute_gt(curve_covars, boxpoints01)
    concatdset = torchdata.ConcatDataset(dsets)
    elem = concatdset[0]
    dT_desired = float(elem["delta_t"][-1] - elem["delta_t"][0]) 
    dataloader = torchdata.DataLoader(concatdset, batch_size=1, shuffle=False)

    # ls = torch.linspace(0.0, 1.0, steps=2).type_as(boxpoints01)
    # gaussianmeans_01 = torch.cat([(1.0-ls[:,None])*boxpoints01[0] + ls[:,None]*(boxpoints01[1]),
    #                             ((1.0-ls[:,None])*boxpoints01[1] + ls[:,None]*boxpoints01[2])[1:],
    #                             ((1.0-ls[:,None])*boxpoints01[2] + ls[:,None]*boxpoints01[3])[1:],
    #                             ((1.0-ls[:,None])*boxpoints01[3] + ls[:,None]*boxpoints01[0])[1:-1]], dim=0)
    gaussianmeans_01 = boxpoints01.clone()
    gaussianmeans_01 = gaussianmeans_01[torch.argsort(torch.atan2(gaussianmeans_01[:,1], gaussianmeans_01[:,0]))]
    gaussianmeans_01 = torch.cat([gaussianmeans_01, torch.zeros_like(gaussianmeans_01[[0,]])], dim=0) 
    # gaussianmeans_01 = torch.cat([gaussianmeans_01, 0.25*(gaussianmeans_01[[0,]] + gaussianmeans_01[[1,]]), 0.75*(gaussianmeans_01[[2,]] + gaussianmeans_01[[3,]])], dim=0)
    # gaussianmeans_01 = torch.cat([gaussianmeans_01, 0.5*(gaussianmeans_01[[0,]] + gaussianmeans_01[[1,]]), 0.5*(gaussianmeans_01[[2,]] + gaussianmeans_01[[3,]])], dim=0)
    # gaussianmeans_01 = torch.cat([gaussianmeans_01, 0.75*(gaussianmeans_01[[0,]] + gaussianmeans_01[[1,]]), 0.75*(gaussianmeans_01[[2,]] + gaussianmeans_01[[3,]])], dim=0)
    
    print(gaussianmeans_01)
    N_gl_sections = 1
    gl1d = GaussLegendre1D(n2, interval=[0, dT_desired/N_gl_sections]).to(dtype=s.dtype, device=s.device)
    
    gl2d = GaussianIntegral2D(n1, intervalx=[-0.5*car_length, 0.5*car_length], intervaly=[-0.5*car_width, 0.5*car_width]).to(dtype=s.dtype, device=s.device)
    circlecheckrad = 1.0*car_diameter
    maxcircle = MaxCircleProb(curve_covars, car_width, car_length).to(tensor=s)
    gl1d = torch.compile(gl1d, fullgraph=True, dynamic=False, backend="inductor", mode="max-autotune-no-cudagraphs")
    gl2d = torch.compile(gl2d, fullgraph=True, dynamic=False, backend="inductor", mode="max-autotune-no-cudagraphs")
    maxcircle = torch.compile(maxcircle, fullgraph=True, dynamic=False, backend="inductor", mode="max-autotune-no-cudagraphs")
    # glcirc = GaussianIntegralCirc(64, radius=circlecheckrad).to(dtype=s.dtype, device=s.device)
    
    s_circlecheck = torch.linspace(0.0, 1.0, steps=2**7).type_as(s)
    M_circlecheck = mu.bezierM(s_circlecheck[None], kbezier)
    # with torch.no_grad():
    _, covars_circlecheck = pbcurve(M_circlecheck)
    covars_circlecheck : torch.Tensor = covars_circlecheck[0]
    print("covars_circlecheck.shape:", covars_circlecheck.shape)
    logtwopi = float(np.log(2.0*np.pi))
    target_stdevs_circlecheck = torch.stack([covars_circlecheck[:,0,0], covars_circlecheck[:,1,1]], dim=-1).sqrt()
    target_logstdevs_circlecheck = (logtwopi + torch.log(target_stdevs_circlecheck).sum(dim=-1))
    target_stdev_inv_matrix_circlecheck = torch.diag_embed(1.0/target_stdevs_circlecheck)
    print("target_logstdevs_circlecheck.shape:", target_logstdevs_circlecheck.shape)
    print("target_stdev_inv_matrix_circlecheck.shape:", target_stdev_inv_matrix_circlecheck.shape)
    etadetached = gl1d.eta.detach().clone()
    # s_gl1d = []
    # M_gl1d = []
    # M_deriv_gl1d = []
    s_gl1d=(torch.stack([etadetached + i*(dT_desired/N_gl_sections) for i in range(N_gl_sections)], dim=0)/dT_desired).ravel()
    # s_gl1d = torch.stack(s_gl1d, dim=0).ravel()
    M_gl1d = mu.bezierM(s_gl1d[None], kbezier)
    M_deriv_gl1d = mu.bezierM(s_gl1d[None], kbezier-1)
    # with torch.no_grad():
    _, covars_gl1d = pbcurve(M_gl1d)
    covars_gl1d : torch.Tensor = covars_gl1d[0]
    print("covars_gl1d.shape:", covars_gl1d.shape)
    target_stdevs = torch.stack([covars_gl1d[:,0,0], covars_gl1d[:,1,1]], dim=-1).sqrt()
    target_logstdevs = (logtwopi + torch.log(target_stdevs).sum(dim=-1))
    target_stdev_inv_matrix = torch.diag_embed(1.0/target_stdevs)
    target_stdev_inv_matrix = target_stdev_inv_matrix.unsqueeze(-3).expand(
            target_stdev_inv_matrix.shape[0], gaussianmeans_01.shape[0], 2, 2)

    # print(s_gl1d)
    # print(target_stdev_inv_matrix.shape)
    # print(target_logstdevs.shape)
    # exit(-1)
    # print(s_gl1d.shape)
    # print(M_gl1d.shape)
    # print(M_deriv_gl1d.shape)

    s_plot = torch.linspace(0.0, 1.0, steps=int(round(1.0*(2**7))), requires_grad=False).type_as(s)
    M_plot = mu.bezierM(s_plot[None], kbezier)
    M_deriv_plot = mu.bezierM(s_plot[None], kbezier-1)
    _, covars_plot = pbcurve(M_plot)
    covars_plot = covars_plot[0]
    print("covars_plot.shape:", covars_plot.shape)
    # print("s_gl1d:", s_gl1d)
    # print("gl2d.eta01:", gl2d.eta_01)
    # logtwopi = float(np.log(2.0*np.pi))

    flip = torch.as_tensor([-1.0, 1.0]).type_as(s_plot)
    weight_individual = (torch.as_tensor(alpha).type_as(s_plot))    
    errors_gl = torch.nan*torch.empty(len(concatdset))
    errors_max = errors_gl.clone()
    errors_max_circle = errors_gl.clone()
    errors_rect = errors_gl.clone()

    comptimes_gl = torch.nan*torch.empty(len(concatdset))
    comptimes_max = comptimes_gl.clone()
    comptimes_max_circle = comptimes_gl.clone()
    comptimes_rect = comptimes_gl.clone()
    comptimes_gt_montecarlo = comptimes_gl.clone()

    thisdir = os.path.normpath(os.path.dirname(__file__))
    outdir = os.path.join(thisdir,"output_plots","collision_checking") if out is None else os.path.join(out, "n1_%d_n2_%d" % (n1, n2))
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    individual_plots_dir = os.path.join(outdir, "individual_plots")
    os.makedirs(individual_plots_dir, exist_ok=False)
    column_width=3.5
    figsize = np.asarray([3.0*column_width, column_width])
    for i, _datadict_ in tqdm.tqdm(enumerate(dataloader), total=len(concatdset)):
        # idx_subselect = torch.arange(0, datadict["attacker_pos"].shape[-2], 1, dtype=torch.int64)
        datadict : dict[str,torch.Tensor] = _datadict_
        # attacker_pos : torch.Tensor = (datadict["attacker_pos"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        # defender_pos : torch.Tensor = (datadict["defender_pos"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        # attacker_vel : torch.Tensor = (datadict["attacker_vel"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        # defender_vel : torch.Tensor = (datadict["defender_vel"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        delta_t : torch.Tensor = datadict["delta_t"].type_as(s)#[:,idx_subselect]
        dT = delta_t[:,-1] - delta_t[:,0]
        
        # _, attacker_curve = bezier.bezierLsqfit(attacker_pos, kbezier, t=delta_t, P0=attacker_pos[:,0], V0=attacker_vel[:,0])#, Vf=attacker_vel[:,-1])   
        
        attacker_curve = (datadict["attacker_curve"][...,[0,1]]).type_as(s)
        attacker_curve_deriv = kbezier*torch.diff(attacker_curve, dim=-2)/dT[0].item()
        # _, defender_curve = bezier.bezierLsqfit(defender_pos, kbezier, t=delta_t, P0=defender_pos[:,0], V0=defender_vel[:,0])#, Vf=defender_vel[:,-1]) 
        defender_curve = (datadict["defender_curve"][...,[0,1]]).type_as(s)
        defender_curve_deriv = kbezier*torch.diff(defender_curve, dim=-2)/dT[0].item()

        attacker_p_plot = (M_plot@attacker_curve)[0]
        attacker_vel_plot = (M_deriv_plot@attacker_curve_deriv)[0]
        attacker_speed_plot = torch.norm(attacker_vel_plot, p=2.0, dim=-1)
        attacker_tau_plot = attacker_vel_plot/attacker_speed_plot[...,None]
        attacker_R_plot = torch.stack([attacker_tau_plot, attacker_tau_plot[...,[1,0]]], dim=-1)
        attacker_R_plot[...,0,1]*=-1.0
        attacker_boxpoints_plot = (attacker_R_plot@boxpoints01.T[None]).transpose(-2,-1) + attacker_p_plot[:,None]

        attacker_p_gl = (M_gl1d@attacker_curve)[0]
        attacker_vel_gl = (M_deriv_gl1d@attacker_curve_deriv)[0]
        attacker_speed_gl = torch.norm(attacker_vel_gl, p=2.0, dim=-1)
        attacker_tau_gl = attacker_vel_gl/attacker_speed_gl[...,None]
        attacker_R_gl = torch.stack([attacker_tau_gl, attacker_tau_gl[...,[1,0]]], dim=-1)
        attacker_R_gl[...,0,1]*=-1.0
        attacker_boxpoints_gl = (attacker_R_gl@boxpoints01.T).transpose(-2,-1) + attacker_p_gl[:,None]

        

        defender_p_plot = (M_plot@defender_curve)[0]
        defender_vel_plot = (M_deriv_plot@defender_curve_deriv)[0]
        defender_speed_plot = torch.norm(defender_vel_plot, p=2.0, dim=-1)
        defender_tau_plot = defender_vel_plot/defender_speed_plot[...,None]
        defender_R_plot = torch.stack([defender_tau_plot, defender_tau_plot[...,[1,0]]], dim=-1)
        defender_R_plot[...,0,1]*=-1.0
  
        tick = time.time()
        defender_Pmean_gl = (M_gl1d@defender_curve)[0]
        defender_vmean_gl = (M_deriv_gl1d@defender_curve_deriv)[0]
        defender_speedmean_gl = torch.norm(defender_vmean_gl, p=2.0, dim=-1)
        defender_taumean_gl = defender_vmean_gl/defender_speedmean_gl[...,None]
        defender_Rmean_gl = torch.stack([defender_taumean_gl, defender_taumean_gl[...,[1,0]]*flip], dim=-1)
        gaussianmeans = (defender_Rmean_gl@gaussianmeans_01.T).transpose(-2,-1) + defender_Pmean_gl[:,None]
        defender_boxpoints_mean_plot = (defender_R_plot@boxpoints01.T).transpose(-2,-1) + defender_p_plot[...,None,:]
        
        # print(gaussianmeans.shape)
        # print(target_stdev_inv_matrix.shape)
        # print(target_logstdevs.shape)
        # print(attacker_p_gl.shape)
        # print(attacker_R_gl.shape)
        (gauss_pts, gaussian_pdf_vals, dense_collision_probs) = gl2d(gaussianmeans[None], target_stdev_inv_matrix[None], target_logstdevs[None],
                                                                       attacker_R_gl[None], attacker_p_gl[None])
        no_collision_probs_rect = torch.prod(1.0 - dense_collision_probs, dim=-1)
        collision_probs_rect = (1.0 - no_collision_probs_rect[0])
        collision_probs_rect_corrected = collision_probs_rect**gamma
        collision_probs_rect_reshape = collision_probs_rect_corrected.view(N_gl_sections, gl1d.eta.shape[0])
        gl1d_integral_rect = torch.sum(torch.stack([
            gl1d(weight_individual*collision_probs_rect_reshape[j] + (1.0-weight_individual)*collision_probs_rect_reshape[j]/(1-collision_probs_rect_reshape[j]))
            # gl1d((collision_probs_rect_reshape[j]/(1-collision_probs_rect_reshape[j]))**gamma)
            for j in range(N_gl_sections)
            ]
        , dim=0))
        pr_nocollision_gl_rect = torch.exp(-gl1d_integral_rect)
        pr_collision_gl_rect = (1.0-pr_nocollision_gl_rect)
        tock = time.time()
        comptimes_rect[i]=(tock-tick)

        if debug:
            print("attacker_p_gl.shape:",  attacker_p_gl.shape)
            print("attacker_R_gl.shape:",  attacker_R_gl.shape)
            print("defender_Pmean_gl.shape:",  defender_Pmean_gl.shape)
            print("defender_Rmean_gl.shape:",  defender_Rmean_gl.shape)
            print("gaussianmeans.shape:",  gaussianmeans.shape)
            print("gauss_pts.shape:",  gauss_pts.shape)
            print("dense_collision_probs.shape:",  dense_collision_probs.shape)
            print("gaussian_pdf_vals.shape:",  gaussian_pdf_vals.shape)
            print("collision_probs_rect.shape:",  collision_probs_rect.shape)
        
        tick = time.time()
        probabilistic_curve : torch.distributions.MultivariateNormal = torch.distributions.MultivariateNormal(defender_curve[0], covariance_matrix=curve_covars, validate_args=False)
        sampled_curves = probabilistic_curve.sample([int(round(1.25*(2**10))),]).type_as(s)
        sampled_curve_derivs = kbezier*torch.diff(sampled_curves,dim=-2)/dT_desired
        tock = time.time()
        time_for_samples = float(tock-tick)
    
        tick = time.time()
        defender_Psamp_gl = M_gl1d@sampled_curves
        defender_vsamp_gl = M_deriv_gl1d@sampled_curve_derivs
        defender_speedsamp_gl = torch.norm(defender_vsamp_gl, p=2.0, dim=-1)
        defender_tausamp_gl = defender_vsamp_gl/defender_speedsamp_gl[...,None]
        defender_Rsamp_gl = torch.stack([defender_tausamp_gl, defender_tausamp_gl[...,[1,0]]*flip], dim=-1)
        boxpoints_samp_gl = (defender_Rsamp_gl@boxpoints01.T[None]).transpose(-2,-1) + defender_Psamp_gl[...,None,:]
        res = cc.rectangle_intersections2(attacker_boxpoints_gl[None].expand_as(boxpoints_samp_gl), boxpoints_samp_gl)
        collision_idx : torch.Tensor = res.collision_idx
        individual_collision_probs : torch.Tensor = ((torch.sum(collision_idx, dim=0)/collision_idx.shape[0]).type_as(s)).clip(0.0,1.0)
        # maxcheck_finalanswer = individual_collision_probs.max()
        # tock = time.time()
        # comptimes_max[i]=(tock-tick) + time_for_samples
        individual_collision_probs_corrected = individual_collision_probs**gamma
        # log_individual_collision_probs = torch.log(individual_collision_probs)
        # individual_nocollision_probs = 1.0 - individual_collision_probs
        # print("individual_collision_probs.shape:", individual_collision_probs.shape)
        individual_collision_probs_reshape=individual_collision_probs_corrected.view(N_gl_sections, gl1d.eta.shape[0])
        # print("individual_collision_probs_reshape.shape:", individual_collision_probs_reshape.shape)
        # gl1d(weight_individual*individual_collision_probs_reshape[j] + (1.0-weight_individual)*individual_collision_probs_reshape[j]/(1-individual_collision_probs_reshape[j]))
           #gl1d(individual_collision_probs_reshape[j]*-torch.log(1.0-individual_collision_probs_reshape[j]))
        gl1d_integral = torch.sum(torch.stack([
            gl1d(weight_individual*individual_collision_probs_reshape[j] + (1.0-weight_individual)*individual_collision_probs_reshape[j]/((1.0-individual_collision_probs_reshape[j])))
            # gl1d((individual_collision_probs_reshape[j]/(1-individual_collision_probs_reshape[j]))**gamma)
            for j in range(N_gl_sections)
            ]
        , dim=0))
        pr_nocollision_gl = torch.exp(-gl1d_integral)
        pr_collision_gl = (1.0-pr_nocollision_gl)
        tock = time.time()
        comptimes_gl[i]=(tock-tick) + time_for_samples

        tick = time.time()
        # attacker_p_circlecheck = (M_circlecheck@attacker_curve)[0]
        # defender_p_circlecheck = (M_circlecheck@sampled_curves)#[0]
        # deltasamp = defender_p_circlecheck - attacker_p_circlecheck#[None]
        # deltasamp_norms = torch.norm(deltasamp, p=2.0, dim=-1)
        # circlecheck_mc_probs = (torch.sum(deltasamp_norms<circlecheckrad, dim=0))/deltasamp_norms.shape[0]
        # circlecheck_final_answer = circlecheck_mc_probs.max()
        circlecheck_final_answer = maxcircle(attacker_curve, defender_curve)
        tock = time.time()
        comptimes_max_circle[i]=(tock-tick) + time_for_samples
        # if debug:
        #     print("deltasamp.shape:",  deltasamp.shape)
        #     print("deltasamp_norms.shape:",  deltasamp_norms.shape)
        #     print("circlecheck_mc_probs.shape:",  circlecheck_mc_probs.shape)


        # tick = time.time()
        # defender_Psamp_plot = M_plot@sampled_curves
        # defender_vsamp_plot = M_deriv_plot@sampled_curve_derivs
        # defender_speedsamp_plot = torch.norm(defender_vsamp_plot, p=2.0, dim=-1)
        # defender_tausamp_plot = defender_vsamp_plot/defender_speedsamp_plot[...,None]
        # defender_Rsamp_plot = torch.stack([defender_tausamp_plot, defender_tausamp_plot[...,[1,0]]*flip], dim=-1)
        # boxpoints_samp_plot = (defender_Rsamp_plot@boxpoints01.T[None]).transpose(-2,-1) + defender_Psamp_plot[...,None,:]
        # res_dense = cc.rectangle_intersections2(attacker_boxpoints_plot[None].expand_as(boxpoints_samp_plot), boxpoints_samp_plot)
        # collision_idx_dense : torch.Tensor = res_dense.collision_idx
        # contain_atleast1_collision = torch.sum(collision_idx_dense, dim=1)>0
        # individual_collision_probs_dense : torch.Tensor = torch.sum(collision_idx_dense, dim=0)/collision_idx_dense.shape[0]
        # maxcheck_finalanswer = individual_collision_probs_dense.max()
        # tock = time.time()
        # comptimes_max[i]=(tock-tick) + time_for_samples
        pr_collision = datadict["gt_prob"].type_as(s)
        individual_collision_probs_dense = datadict["individual_collision_probs_dense"].type_as(s)
        maxcheck_finalanswer = individual_collision_probs_dense.max()

        # tick = time.time()
        # pr_collision = contain_atleast1_collision.sum()/contain_atleast1_collision.shape[0]
        # tock = time.time()
        # # tock = time.time()
        # comptimes_gt_montecarlo[i]=(tock-tick) + comptimes_max[i]

        if debug:
            # print("res_ednse.solution.shape:", res_dense.solution.shape)
            # print("collision_idx_dense.shape:", collision_idx_dense.shape)
            print("pr_collision:", pr_collision)
            print("pr_collision_gl:", pr_collision_gl)
            print("pr_collision_gl_rect:", pr_collision_gl_rect)
            print("individual_collision_probs.max():", individual_collision_probs.max())

        errors_gl[i] = (pr_collision_gl- pr_collision).type_as(errors_gl).nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
        errors_rect[i] = (pr_collision_gl_rect - pr_collision).type_as(errors_rect).nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
        errors_max[i] = (maxcheck_finalanswer - pr_collision).type_as(errors_max).nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)
        errors_max_circle[i] = (circlecheck_final_answer - pr_collision).type_as(errors_max_circle).nan_to_num(nan=1.0, posinf=1.0, neginf=1.0)

        # if (save=="all") or debug:
        #     fig , ax = plt.subplots(nrows=1, ncols=1, num="Yay", frameon=False, layout="constrained",figsize=figsize.tolist())
        #     fig_time , ax_tuple = plt.subplots(nrows=1, ncols=2, num="Yay Time", frameon=False, layout="constrained", figsize=figsize.tolist())
        #     ax_time : matplotlib.axes.Axes = ax_tuple[0]
        #     circlevalsplot, = ax_time.plot((s_circlecheck*dT_desired).cpu().numpy(), circlecheck_mc_probs.cpu().numpy(), label="Max Circle")
        #     densemcvalsplot, = ax_time.plot((s_plot*dT_desired).cpu().numpy(), individual_collision_probs_dense.cpu().numpy(), label="Max Rect")
        #     # ax_time.plot((s_circlecheck*dT_desired).cpu().numpy(), cdfvals_circle.cpu().numpy(), label="Circle Approximation")
        #     glrectvalsplot, = ax_time.plot((s_gl1d*dT_desired).cpu().numpy(), collision_probs_rect.cpu().numpy(), label="GLR")
        #     qmlglvalsplot, = ax_time.plot((s_gl1d*dT_desired).cpu().numpy(), individual_collision_probs.cpu().numpy(), label="QMLGL")
        #     qmlglvalsplot.set_visible(False)
        #     ax_time.set_ylim(0.0, 1.25)
        #     ax_time.yaxis.set_ticks(np.linspace(0.0, 1.0, num=5))

        #     # attacker_pfit_line, = ax.plot(attacker_pos[0,:,0], attacker_pos[0,:,1], label="Attacker Pfit", linestyle="--")

        #     with plt.rc_context({"text.usetex" : True}):
        #         defender_curve_plot, = ax.plot(*(defender_p_plot.T.cpu().numpy()), label=r"$\mathcal{T}_{target}$", color=utils.color.UVA_BLUE)
        #         defender_cpoints_scatter= ax.scatter(*(defender_curve[0].T.cpu().numpy()), label=r"$\mathcal{C}_{i,target}$", color=defender_curve_plot.get_color(), s=2**1.5)
        #         attacker_curve_plot, = ax.plot(*(attacker_p_plot.T.cpu().numpy()), label=r"$\mathcal{T}_{ego}$", color=utils.color.UVA_ORANGE)  
        #         attacker_cpoints_scatter = ax.scatter(*(attacker_curve[0].T.cpu().numpy()), label=r"$\mathbf{C}_{i,ego}$",  color=attacker_curve_plot.get_color(), s=float(defender_cpoints_scatter.get_sizes()[0]))
            
        #     # defender_pfit_line, = ax.plot(defender_pos[0,:,0], defender_pos[0,:,1], label="Defender Pfit", linestyle="--")
        #     idx_select_curves=np.random.choice(np.arange(0, defender_Psamp_plot.shape[0], step=1, dtype=np.int64), replace=False, size=int(round(0.02*defender_Psamp_plot.shape[0])))
        #     for j in range(0, idx_select_curves.shape[0]):
        #         ax.plot(*(defender_p_circlecheck[idx_select_curves[j]].T.cpu().numpy()), color=defender_curve_plot.get_color(), alpha=0.075)
        #     ratios = np.linspace(0.01, 2.0,  num=20) 
        #     alphas = np.linspace(1.0,  0.01, num=ratios.shape[0])**1.25
        #     for j in range(curve_covars.shape[0]):
        #         center = defender_curve[0,j].cpu().numpy()
        #         stdev = curve_covars[j,0,0].sqrt().item()
        #         for k in range(ratios.shape[0]):
        #             circle : matplotlib.patches.Circle = ax.add_patch(matplotlib.patches.Circle(center, radius=ratios[k]*stdev, fill=False, edgecolor=defender_curve_plot.get_color(), alpha=alphas[k]))

        #     idx_plot_boxpoints = int(round(0.5*boxpoints_samp_plot.shape[1]))
        #     Sigma1 = covars_plot[idx_plot_boxpoints]
        #     mu1 = defender_boxpoints_mean_plot[idx_plot_boxpoints]
        #     mvn1 = torch.distributions.MultivariateNormal(mu1, covariance_matrix=Sigma1[None].expand(4,2,2).clone())
        #     # rectpoints_samp = mvn1.sample([2**10,])

        #     Sigma0 = torch.stack([torch.cov(boxpoints_samp_plot[...,idx_plot_boxpoints,j,:].T) for j in range(boxpoints_samp_plot.shape[-2])], dim=0)
        #     mu0 = torch.mean(boxpoints_samp_plot[:,idx_plot_boxpoints], dim=0)
        #     mvn0 = torch.distributions.MultivariateNormal(mu0, covariance_matrix=Sigma0)

        #     kl = torch.distributions.kl_divergence(mvn0, mvn1)
        #     if debug:
        #         print("KL divergence from sampled corners to theoretical corners:", kl)


        #     ax.set_aspect(aspect="equal", adjustable='box')

        #     abserrors=[]
        #     colors=[]
        #     labels=[]
        #     for line, value, error in [(qmlglvalsplot, pr_collision_gl.item(), errors_gl[i].item()), 
        #                                (glrectvalsplot, pr_collision_gl_rect.item(), errors_rect[i].item()), 
        #                                (densemcvalsplot, maxcheck_finalanswer.item(), errors_max[i].item()), 
        #                                (circlevalsplot, circlecheck_mc_probs.max().item(), errors_max_circle[i].item())]:
        #         ax_time.axhline(y=value, color=line.get_color(), linestyle="--")
        #         abserrors.append(float(np.abs(error)))
        #         colors.append(line.get_color())
        #         labels.append(line.get_label())
        #     ax_barchart : matplotlib.axes.Axes = ax_tuple[1]
        #     ax_time.axhline(y=pr_collision.item(), color="black", linestyle="--")
        #     ax_barchart.bar(np.arange(1, len(abserrors)+1, step=1, dtype=np.int64), abserrors, label=labels, color=colors)
        #     # legend = ax_time.legend(frameon=False, loc=[0.05, 1.0/1.25])
        #     if i==0 and (save is not None) and (not len(save)==0):
        #         with plt.rc_context({"text.usetex" : True}):
        #             fig_legend, ax_legend, bbox_legend = utils.export_legend(ax)
        #             fig_legend.savefig(os.path.join(individual_plots_dir, "example.legend.svg"), transparent=True, bbox_inches=bbox_legend)
        #             plt.close(fig=fig_legend)
        #             fig_legend, ax_legend, bbox_legend = utils.export_legend(ax_time)
        #             fig_legend.savefig(os.path.join(individual_plots_dir, "example_time.legend.svg"), transparent=True, bbox_inches=bbox_legend)
        #             plt.close(fig=fig_legend)
        #     fig.draw(renderer=fig.canvas.get_renderer()) 
        #     # figinches_to_display = fig.dpi_scale_trans
        #     # display_to_figinches = figinches_to_display.inverted()

        #     # fig01_to_display = fig.transFigure
        #     # display_to_fig01 = fig01_to_display.inverted()

        #     # fig01_to_figinches=fig01_to_display + display_to_figinches
        #     ax.xaxis.set_ticks([])
        #     ax.yaxis.set_ticks([])
        #     for k in ax.spines.keys():
        #         ax.spines[k].set_visible(False)
        #     if (save=="all"):
        #         fig.savefig(os.path.join(individual_plots_dir, "example_%d.svg" % (i,)), transparent=True, pad_inches=0.0)
        #         p = PdfPages(os.path.join(individual_plots_dir, "example_%d.full.pdf" % (i,))) 
        #         fig.savefig(p, format="pdf", transparent=True)
        #         fig_time.savefig(p, format="pdf", transparent=True)
        #         p.close()
        #         fig_time.savefig(os.path.join(individual_plots_dir, "example_%d.time.svg" % (i,)), transparent=True, pad_inches=0.0)
        #     plt.show() if debug else plt.close('all')
            
    bins=50
    fig_hist, ax_hist = plt.subplots(frameon=False, layout="constrained")
    ax_hist.hist(errors_gl.cpu().numpy().tolist(), bins=bins, label="QMLGL")
    ax_hist.hist(errors_rect.cpu().numpy().tolist(), bins=bins, label="GLR")
    ax_hist.hist(errors_max.cpu().numpy().tolist(), bins=bins, label="Max")
    ax_hist.hist(errors_max_circle.cpu().numpy().tolist(), bins=bins, label="Max Circle")
    ax_hist.legend()
    fig_hist.savefig(os.path.join(outdir, "combined.histogram.svg"), transparent=True, pad_inches=0.0)
    plt.close(fig=fig_hist)
    summary : dict = {
        "sigma0" : sigma0,
        "sigma1" : sigma1,
        "datadir" : datadir,
        "kbezier": kbezier,
        "gamma": gamma,
        "alpha": alpha,
        "n1": n1,
        "n2": n2,
        "controlpoint_sigmas" : curve_covars.cpu().numpy().tolist(),
    }
    quantiles = torch.as_tensor([0.05, 0.95])
    relerrors=[]
    abserrors=[]
    names = []
    dense_errors : dict[str,np.ndarray] = dict()
    dense_comptimes : dict[str,np.ndarray] = {"Dense Monte Carlo" : comptimes_gt_montecarlo.cpu().numpy()}
    for name, errors, comptimes in [("QMLGL", errors_gl, comptimes_gl), ("GLR", errors_rect, comptimes_rect), ("Max", errors_max, comptimes_max), ("Max_Circle", errors_max_circle, comptimes_max_circle)]:
        dense_errors[name]=errors.cpu().numpy().copy()
        dense_comptimes[name]=comptimes.cpu().numpy().copy()
        print("Mean Abs Error %s: " % (name,), errors.abs().mean().item())
        relerrors.append(errors.cpu().numpy())
        abserrors.append(errors.abs().cpu().numpy())
        names.append(name)
        kde : scipy.stats.gaussian_kde = scipy.stats.gaussian_kde(errors.cpu().numpy())
        error_plot = np.linspace(errors.min().item(), errors.max().item(), num=120)
        quantile_values = torch.quantile(errors, quantiles.type_as(errors))
        overestimating_idx = errors>=0
        underestimating_idx = errors<0
        overestimating_percentage = (overestimating_idx.sum()/errors.shape[0]).item()
        underestimating_percentage = (underestimating_idx.sum()/errors.shape[0]).item()
        pdfvals_plot = kde.pdf(error_plot)
        fig, ax = plt.subplots(frameon=False, layout="constrained")
        ax.set_xlim(-1.0, 1.2)
        ax.plot(error_plot, pdfvals_plot)
        vline0 = ax.axvline(x=quantile_values[0].item(), linestyle="--", color="black")
        vline1 = ax.axvline(x=quantile_values[1].item(), linestyle=vline0.get_linestyle(), color=vline0.get_color())
        fig.savefig(os.path.join(outdir, "%s.kde.svg" % (name,)), transparent=True, pad_inches=0.0)
        plt.close(fig=fig)
        fig_hist, ax_hist = plt.subplots(frameon=False, layout="constrained")
        ax_hist.hist(errors.cpu().numpy().tolist(), bins=bins)
        fig_hist.savefig(os.path.join(outdir, "%s.histogram.svg" % (name,)), transparent=True, pad_inches=0.0)
        plt.close(fig=fig_hist)
        currentabserrors = errors.abs()
        summary[name] = {
            "comptime_mean" : comptimes[5:].mean().item(),
            "comptime_median" : comptimes[5:].median().item(),
            "looprate_mean" : (1.0/comptimes[5:]).mean().item(),
            "looprate_median" : (1.0/comptimes[5:]).median().item(),
            "MAE" : currentabserrors.mean().item(),
            "MAE_stdev" : currentabserrors.std().item(),
            "MAE_over" : currentabserrors[overestimating_idx].mean().item(),
            "MAE_under" : currentabserrors[underestimating_idx].mean().item(),
            "Overestimating Percentage" : overestimating_percentage,
            "Underestimating Percentage" : underestimating_percentage,
            "Fifth Percentile" : quantile_values[0].item(),
            "Ninetyfifth Percentile" : quantile_values[1].item()
            # "percentiles" : {
            #     "%2.2f" % (quantiles[0].item()*100.0,) : quantile_values[0].item(),
            #     "%2.2f" % (quantiles[1].item()*100.0,)  : quantile_values[1].item()
            # }
        }
    summary["Dense Monte Carlo"] = {
        "MAE" : 0.0,
        "Overestimating Percentage" : 0.0,
        "Underestimating Percentage" : 0.0,
        "Fifth Percentile" : 0.0,
        "Ninetyfifth Percentile" : 0.0,
        "comptime_mean" : comptimes_gt_montecarlo.mean().item(),
        "comptime_median" : comptimes_gt_montecarlo.median().item(),
    }
    fig_box, ax_box = plt.subplots(frameon=False, layout="constrained", figsize=figsize)
    ax_box.boxplot(abserrors, patch_artist=True, tick_labels=names)
    fig_box.savefig(os.path.join(outdir, "combined.absolute.boxplot.svg"), transparent=True, pad_inches=0.0)
    plt.close(fig=fig_box)
    
    fig_box, ax_box = plt.subplots(frameon=False, layout="constrained", figsize=figsize)
    ax_box.boxplot(relerrors, patch_artist=True, tick_labels=names)
    fig_box.savefig(os.path.join(outdir, "combined.relative.boxplot.svg"), transparent=True, pad_inches=0.0)
    plt.close(fig=fig_box)
    with open(os.path.join(outdir, "dense_comptimes.npz"), "wb") as f:
        np.savez(f, **dense_comptimes)
    with open(os.path.join(outdir, "dense_errors.npz"), "wb") as f:
        np.savez(f, **dense_errors)
    with open(os.path.join(outdir, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, stream=f)
    with open(os.path.join(outdir, "INDIVIDUAL_GLQ_TEST"), "w") as f:
        f.write("Yay\n")
        
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Test the collision detection algorithm")
    parser.add_argument("datadir", type=str, help="The dataset to test on")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--sigma1", type=float, default=1.0)
    parser.add_argument("--kbezier", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--car-length", type=float, default=5.2)
    parser.add_argument("--car-width", type=float, default=2.0)
    parser.add_argument("--n1", type=int, default=12)
    parser.add_argument("--n2", type=int, default=24)
    main(**(vars(parser.parse_args())))