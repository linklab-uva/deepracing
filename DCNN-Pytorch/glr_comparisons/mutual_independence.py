import deepracing, deepracing_models.math_utils as mu, deepracing_models.math_utils.bezier as bezier
import deepracing_models.data_loading.file_datasets.OvertakingTrajectoriesDataset as otd
import deepracing_models.math_utils.rotations as drrot
import numpy as np
import os
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
import shapely, shapely.geometry
from deepracing_models.math_utils.statistics import CollisionProbabilityEstimator
from deepracing_models.math_utils.integrate import GaussLegendre1D, GaussianIntegral2D, GaussianIntegralCirc
import deepracing_models.math_utils.collision_checking as cc
from deepracing_models.probabilistic_models import ProbabilisticBezierCurve
import scipy.stats
from matplotlib.colors import TABLEAU_COLORS, same_color
import shutil
import utils
from glr_comparisons.results_aggregation import save_results

class MutualIndependence(torch.nn.Module):
    def __init__(self, curve_covars : torch.Tensor, car_width : float, car_length : float):
        super(MutualIndependence, self).__init__()
        kbezier = curve_covars.shape[0] - 1
        self.s_eval = torch.nn.Parameter(torch.linspace(0.0, 1.0, steps=10).double(), requires_grad=False)
        M_eval = bezier.bezierM(self.s_eval[None].detach(), kbezier)
        covars_eval = torch.zeros([self.s_eval.shape[0], 2, 2]).type_as(curve_covars)
        for i in range(covars_eval.shape[0]):
            for j in range(kbezier+1):
                covars_eval[i]+=(M_eval[0,i,j]**2)*(curve_covars[j])
        self.covars_eval = torch.nn.Parameter(covars_eval.cpu().double(), requires_grad=False)
        self.curve_covars = torch.nn.Parameter(curve_covars.cpu().double(), requires_grad=False)
        car_diameter = float(np.sqrt(car_width**2 + car_length**2))
        car_radius = 0.5*car_diameter
        self.min_long_separation = torch.nn.Parameter(torch.as_tensor(0.5*car_length + car_radius).double(), requires_grad=False)
        self.min_lat_separation = torch.nn.Parameter(torch.as_tensor(0.5*car_width + car_radius).double(), requires_grad=False)
        self.flip =  torch.nn.Parameter(torch.as_tensor([-1.0,1.0]).double(), requires_grad=False)
    def forward(self, attacker_curve, defender_curve):
        kbezier = self.curve_covars.shape[0]-1
        M_eval = bezier.bezierM(self.s_eval[None], kbezier)
        M_eval_deriv = bezier.bezierM(self.s_eval[None], kbezier-1)
        attacker_pos = (M_eval@attacker_curve)[0]

        defender_curve_deriv = kbezier*torch.diff(defender_curve, dim=-2)
        defender_pos = (M_eval@defender_curve)[0]
        defender_vel = (M_eval_deriv@defender_curve_deriv)[0]
        defender_speeds = torch.linalg.vector_norm(defender_vel, dim=-1)
        defender_tau = defender_vel/defender_speeds[...,None]
        defender_nu = defender_tau[...,[1,0]]*self.flip
        deltas = attacker_pos - defender_pos
        long_distances = torch.linalg.vecdot(deltas, defender_tau, dim=-1)
        lat_distances = torch.linalg.vecdot(deltas, defender_nu, dim=-1)

        no_collision_prob = torch.ones_like(long_distances[0])
        for j in range(attacker_pos.shape[0]):
            stdev = torch.sqrt(self.covars_eval[j,0,0])
            long_distance_rv= torch.distributions.Normal(long_distances[j], stdev)
            lat_distance_rv = torch.distributions.Normal(lat_distances[j], stdev)

            prob_long_overlap = long_distance_rv.cdf(self.min_long_separation) - long_distance_rv.cdf(-self.min_long_separation)
            prob_lat_overlap = lat_distance_rv.cdf(self.min_lat_separation) - lat_distance_rv.cdf(-self.min_lat_separation)
            prob_both_overlap = prob_long_overlap*prob_lat_overlap
            no_collision_prob*=(1.0 - prob_both_overlap)


        return (1.0 - no_collision_prob).clip(min=0.0, max=1.0)

def main(*args, datadir : str, debug : bool, save : str | None, out : str | None, 
             car_length : float, car_width : float, kbezier: int, sigma0 : float, sigma1 : float):
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
    s = torch.linspace(0.0, 1.0, steps=60, dtype=torch.float64)
    try:
        s = s.cuda(0)
    except:
        pass
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
    curve_covars = torch.stack([pbcurve.covariance[i] for i in range(kbezier+1)]).detach().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).type_as(s)
    model = torch.compile(MutualIndependence(curve_covars, car_width, car_length).to(tensor=s),
                            fullgraph=True, dynamic=False, backend="inductor", mode="max-autotune-no-cudagraphs")
    boxpoints01 = (torch.stack(torch.meshgrid([
        0.5*torch.linspace(-car_length, car_length, steps=2),
        0.5*torch.linspace(-car_width, car_width, steps=2)
    ], indexing="ij"), dim=0)).reshape(2,-1).T.type_as(s)
    boxpoints01 = boxpoints01[torch.argsort(torch.atan2(boxpoints01[:,1], boxpoints01[:,0]))]
    for dset in dsets:
        dset.compute_gt(curve_covars, boxpoints01)
    concatdset = torchdata.ConcatDataset(dsets)
    dataloader = torchdata.DataLoader(concatdset, batch_size=1, shuffle=False)
    
    
    flip = torch.as_tensor([-1.0, 1.0]).type_as(s)

    s_eval = torch.linspace(0.0, 1.0, steps=10).type_as(s)
    M_eval = bezier.bezierM(s_eval[None], kbezier)
    M_eval_deriv = bezier.bezierM(s_eval[None], kbezier-1)
    
    covars_eval = torch.zeros([s_eval.shape[0], 2, 2]).type_as(s)
    for i in range(covars_eval.shape[0]):
        for j in range(kbezier+1):
            covars_eval[i]+=(M_eval[0,i,j]**2)*curve_covars[j]
    predictions = torch.empty(len(concatdset), dtype=torch.float64)
    ground_truths = predictions.clone()
    comptimes = predictions.clone()
    
    for i, _datadict_ in tqdm.tqdm(enumerate(dataloader), total=len(concatdset)):
        datadict : dict[str,torch.Tensor] = _datadict_
        # idx_subselect = torch.arange(0, datadict["attacker_pos"].shape[-2], 1, dtype=torch.int64)
        ground_truths[i] = (datadict["gt_prob"]).type_as(ground_truths)


        # Mfit, attacker_curve = bezier.bezierLsqfit(attacker_pos, kbezier, t=sfit[None])
        # _, defender_curve = bezier.bezierLsqfit(defender_pos, kbezier, M=Mfit)

        attacker_curve = (datadict["attacker_curve"][...,[0,1]]).type_as(s)
        defender_curve = (datadict["defender_curve"][...,[0,1]]).type_as(s)

        tick = time.time()
        predictions[i] = model(attacker_curve, defender_curve).type_as(predictions)
        tock = time.time()
        comptimes[i] = tock - tick
    
    # maevals = (predictions - ground_truths).abs()
    # overestimating_idx = (predictions>ground_truths)
    # comptimes = comptimes[5:]
    # print("MAE:", maevals.mean())
    # print("Mean computation time:", comptimes.mean())
    # print("Mean loop rate:", (1.0/comptimes).mean())
    # print("Median computation time:", comptimes.median())
    # print("Median loop rate:", (1.0/comptimes).median())
    # print("Overestimating percentage:", (overestimating_idx.sum()/overestimating_idx.shape[0]))
    if out is not None:
        resultsout = os.path.abspath(os.path.join(out, "Mutual_Independence"))
        summary = save_results(resultsout, predictions, ground_truths, comptimes)
        print(summary)
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser("Test the risk density algorithm")
    parser.add_argument("datadir", type=str, help="The dataset to test on")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--sigma0", type=float, default=0.1)
    parser.add_argument("--sigma1", type=float, default=1.0)
    parser.add_argument("--car-length", type=float, default=5.2)
    parser.add_argument("--car-width", type=float, default=2.0)
    parser.add_argument("--kbezier", type=int, default=5)
    main(**(vars(parser.parse_args())))