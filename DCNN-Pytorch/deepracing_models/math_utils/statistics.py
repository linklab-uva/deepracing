import torch, torch.nn
def gaussian_pdf(means : torch.Tensor, covarmats : torch.Tensor, eval : torch.Tensor):
    batchdims_means = means.shape[:-1]
    batchdims_eval = eval.shape[:-1]
    batchdims_covars = covarmats.shape[:-2]
    if not (batchdims_means==batchdims_covars==batchdims_eval):
        raise ValueError("Got inconsistent batch dimensions. means have batch dims %s covars have batch dims %s. eval points have batch dims %s" % (str(batchdims_means), str(batchdims_covars), str(batchdims_eval)))
    if not (means.shape[-1]==eval.shape[-1]==covarmats.shape[-2]==covarmats.shape[-1]):
        raise ValueError("Got inconsistent point dimensions. means has final dimension %d. eval points have final dimension %d. covarmats has final 2 dimensions %dX%d" % (means.shape[-1],eval.shape[-1],covarmats.shape[-2],covarmats.shape[-1]))
    eighresult = torch.linalg.eigh(covarmats)
    eigenvalues : torch.Tensor = eighresult[0]
    eigenvectors : torch.Tensor = eighresult[1]
    eigenvalues_inv = 1.0/eigenvalues
    inv_sqrt_determinants = torch.sqrt(torch.prod(eigenvalues_inv, dim=-1))
    constant_factors = 0.159154943*torch.ones_like(inv_sqrt_determinants) if means.shape[-1]==2 else torch.pow(torch.as_tensor(6.283185307), -0.5*means.shape[-1]).item()*torch.ones_like(inv_sqrt_determinants)
    precision_mats = eigenvectors @ torch.diag_embed(eigenvalues_inv) @ eigenvectors.transpose(-2,-1)
    deltas = eval - means
    mahalanobis = -0.5*torch.sum(deltas*(precision_mats @ deltas[...,None]).squeeze(-1), dim=-1)
    return constant_factors*inv_sqrt_determinants*torch.exp(mahalanobis)

def cov(m, rowvar=True, inplace=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
from deepracing_models.math_utils.integrate import GaussLegendre1D, GaussianIntegral2D
from deepracing_models.math_utils.bezier import compositeBezierEval
import torch.distributions
class CollisionProbabilityEstimator(torch.nn.Module):
    def __init__(self, gauss_order_time : int, dT : float, gauss_order_space : int, lat_buffer : float, long_buffer : float, requires_grad=False) -> None:
        super(CollisionProbabilityEstimator, self).__init__()
        self.gl1d : GaussLegendre1D = GaussLegendre1D(gauss_order_time, interval=(0, dT), requires_grad=requires_grad)
        self.gaussian_pdf_integrator : GaussianIntegral2D = GaussianIntegral2D(gauss_order_space, requires_grad=requires_grad, intervalx=[-long_buffer, long_buffer], intervaly=[-lat_buffer, lat_buffer])
        self.tangent_to_normal_rotmat : torch.nn.Parameter = torch.nn.Parameter(torch.as_tensor([
            [ 0.0, 1.0],
            [-1.0, 0.0]
        ]), requires_grad=requires_grad)
    # def forward(self, candidate_bezier_curves : torch.Tensor, candidate_curve_tstart : torch.Tensor, candidate_curve_dt : torch.Tensor, **kwargs):
    def forward(self, *args):
        candidate_bezier_curves : torch.Tensor = args[0]
        candidate_curve_tstart : torch.Tensor = args[1]
        candidate_curve_dt : torch.Tensor = args[2]
        if len(args)==4:
            mvn : torch.distributions.MultivariateNormal = args[3]
        else:
            mvn = torch.distributions.MultivariateNormal(args[3], covariance_matrix=args[4], validate_args=False)
        batchdims = list(candidate_curve_dt.shape[:-1])
        nbatchdims = candidate_curve_dt.ndim - 1
        time_integral_nodes = self.gl1d.eta.view([1 for _ in range(nbatchdims)] + [self.gl1d.eta.shape[0],]).expand(batchdims + [self.gl1d.eta.shape[0],]) + candidate_curve_tstart[:,[0,]]
        # for _ in range(nbatchdims):
        #     time_integral_nodes = time_integral_nodes[None]
        # time_integral_nodes = time_integral_nodes.expand(batchdims + [self.gl1d.eta.shape[0],]) + candidate_curve_tstart[:,[0,]]
        candidate_curve_points, idxbuckets = compositeBezierEval(candidate_curve_tstart, candidate_curve_dt, candidate_bezier_curves, time_integral_nodes)
        candidate_curve_derivs : torch.Tensor = (candidate_bezier_curves.shape[-2]-1)*(candidate_bezier_curves[...,1:,:] - candidate_bezier_curves[...,:-1,:])/candidate_curve_dt[...,None,None]
        candidate_curve_vels, _ = compositeBezierEval(candidate_curve_tstart, candidate_curve_dt, candidate_curve_derivs, time_integral_nodes, idxbuckets=idxbuckets)
        candidate_curve_tangents = candidate_curve_vels/torch.norm(candidate_curve_vels, p=2.0, dim=-1, keepdim=True)
        candidate_curve_normals = candidate_curve_tangents @ self.tangent_to_normal_rotmat
        candidate_curve_rotmats = torch.stack([candidate_curve_tangents, candidate_curve_normals], dim=-1)
        gauss_pts, gaussian_pdf_vals, collision_probs = self.gaussian_pdf_integrator(mvn, candidate_curve_rotmats, candidate_curve_points)
        overall_lambdas = self.gl1d(collision_probs)
        overall_collision_free_probs = torch.exp(-overall_lambdas)
        return gauss_pts, gaussian_pdf_vals, mvn, collision_probs, overall_lambdas, overall_collision_free_probs
