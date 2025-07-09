import os, sys
thisdir=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(thisdir, "..")))
import deepracing_models.math_utils as mu, deepracing_models.math_utils.bezier as bezier
import deepracing_models.data_loading.file_datasets.OvertakingTrajectoriesDataset as otd
import deepracing_models.math_utils.rotations as drrot
import numpy as np
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
from deepracing_models.math_utils.statistics import CollisionProbabilityEstimator
from deepracing_models.math_utils.integrate import GaussLegendre1D, GaussianIntegral2D, GaussianIntegralCirc
import deepracing_models.math_utils.collision_checking as cc
from deepracing_models.probabilistic_models import ProbabilisticBezierCurve
from glr_comparisons.results_aggregation import save_results
class RiskDensity(torch.nn.Module):
    def __init__(self, curve_covars : torch.Tensor, gauss_order : int, car_width : float, car_length : float):
        super(RiskDensity, self).__init__()
        car_diameter = float(np.sqrt(car_width**2 + car_length**2))
        self.car_diameter = torch.nn.Parameter(torch.as_tensor(car_diameter).double(), requires_grad=False)
        covars_eval = torch.zeros([gauss_order, 2, 2]).type_as(curve_covars)
        kbezier=curve_covars.shape[0]-1
        eta11, weights11 = (torch.as_tensor(v).double() for v in np.polynomial.legendre.leggauss(gauss_order))
        self.weights = torch.nn.Parameter(0.5*weights11, requires_grad=False)
        self.s_eval = torch.nn.Parameter(0.5*eta11 + 0.5, requires_grad=False)
        M_eval = bezier.bezierM(self.s_eval[None].detach(), kbezier).type_as(curve_covars)
        for i in range(gauss_order):
            for j in range(kbezier+1):
                covars_eval[i]+=(M_eval[0,i,j]**2)*curve_covars[j]
        self.curve_covars = torch.nn.Parameter(curve_covars.cpu().double(), requires_grad=False)
        self.covars_eval = torch.nn.Parameter(covars_eval.cpu().double(), requires_grad=False)
    def forward(self, attacker_curve : torch.Tensor, defender_curve : torch.Tensor):
        kbezier=self.curve_covars.shape[0]-1
        M_eval = bezier.bezierM(self.s_eval[None], kbezier)
        M_eval_deriv = bezier.bezierM(self.s_eval[None], kbezier-1)
        attacker_eval = (M_eval[None] @ attacker_curve)[0]
        attacker_curve_deriv = kbezier*torch.diff(attacker_curve, dim=-2)
        attacker_curve_deriv_eval = (M_eval_deriv[None] @ attacker_curve_deriv)[0]
        defender_eval = (M_eval[None] @ defender_curve)[0]
        mvn_eval = torch.distributions.MultivariateNormal(defender_eval, covariance_matrix=self.covars_eval)
        log_pdf_vals = mvn_eval.log_prob(attacker_eval)
        pdf_vals =  log_pdf_vals.exp()
        risk_density_vals =  torch.linalg.vector_norm(attacker_curve_deriv_eval, dim=-1)*pdf_vals 
        return (2.0*self.car_diameter*torch.sum(self.weights*risk_density_vals)).clip(0.0, 1.0)
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

    gauss_order = 32
    model = torch.compile(RiskDensity(curve_covars, gauss_order, car_width, car_length).to(tensor=s),
                        fullgraph=True, dynamic=False, backend="inductor", mode="max-autotune-no-cudagraphs")
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
    car_diameter = float(np.linalg.norm(np.asarray([car_length, car_width]), ord=2.0, axis=0))
    eta11, weights11 = (torch.as_tensor(v).type_as(s) for v in np.polynomial.legendre.leggauss(gauss_order))
    weights = 0.5*weights11
    s_eval = 0.5*eta11 + 0.5
   
    M_eval = mu.bezierM(s_eval[None], kbezier)[0]
    M_eval_deriv = mu.bezierM(s_eval[None], kbezier-1)[0]
    # print(M_eval.shape)
    # print(curve_covars.shape)
    covars_eval = torch.zeros([gauss_order, 2, 2]).type_as(s)
    for i in range(gauss_order):
        for j in range(kbezier+1):
            covars_eval[i]+=(M_eval[i,j]**2)*curve_covars[j]
    # print(covars_eval)
    # s_plot = torch.linspace(0.0, 1.0, steps=int(round(1.0*(2**7))), requires_grad=False).type_as(s)
    # M_plot = mu.bezierM(s_plot[None], kbezier)
    # M_deriv_plot = mu.bezierM(s_plot[None], kbezier-1)
    # mvn_eval = torch.distributions.MultivariateNormal(torch.zeros(gauss_order,2).type_as(s), covariance_matrix=covars_eval)
    
    predictions = torch.empty(len(concatdset), dtype=torch.float64)
    ground_truths = predictions.clone()
    comptimes = predictions.clone()
    for i, _datadict_ in tqdm.tqdm(enumerate(dataloader), total=len(concatdset)):
        datadict : dict[str,torch.Tensor] = _datadict_
        # idx_subselect = torch.arange(0, datadict["attacker_pos"].shape[-2], 1, dtype=torch.int64)

        attacker_pos : torch.Tensor = (datadict["attacker_pos"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        defender_pos : torch.Tensor = (datadict["defender_pos"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        attacker_vel : torch.Tensor = (datadict["attacker_vel"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        defender_vel : torch.Tensor = (datadict["defender_vel"][...,[0,1]]).type_as(s)#[:,idx_subselect]
        ground_truths[i] = (datadict["gt_prob"]).type_as(ground_truths)

        delta_t : torch.Tensor = datadict["delta_t"].type_as(s)#[:,idx_subselect]
        dT = delta_t[:,-1] - delta_t[:,0]
        sfit = torch.linspace(0.0, 1.0, steps=attacker_pos.shape[1]).type_as(s)

        # Mfit, attacker_curve = bezier.bezierLsqfit(attacker_pos, kbezier, t=sfit[None])
        # _, defender_curve = bezier.bezierLsqfit(defender_pos, kbezier, M=Mfit)
        attacker_curve = (datadict["attacker_curve"][...,[0,1]]).type_as(s)
        defender_curve = (datadict["defender_curve"][...,[0,1]]).type_as(s)
        # print(sfit)
        tick = time.time()
        predictions[i] = model(attacker_curve, defender_curve).type_as(predictions)
        tock = time.time()
        # print("risk_density_collision_prob:", risk_density_collision_prob)
        comptimes[i] = tock - tick
        # maevals[i] = torch.abs(pr_collision - risk_density_collision_prob).type_as(maevals)
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
        resultsout = os.path.abspath(os.path.join(out, "Risk_Density"))
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
    parser.add_argument("--kbezier", type=int, default=7)
    main(**(vars(parser.parse_args())))