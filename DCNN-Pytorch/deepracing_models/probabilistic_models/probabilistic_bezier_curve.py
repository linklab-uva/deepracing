from sklearn import manifold
import torch, torch.nn, torch.nn.parameter
import deepracing_models.math_utils
import geotorch, geotorch.parametrize
import geoopt
class ProbabilisticBezierCurve(torch.nn.Module):
    def __init__(self, mean : torch.Tensor, covariance : torch.Tensor):
        super(ProbabilisticBezierCurve, self).__init__()
        self.mean : torch.nn.ParameterList = torch.nn.ParameterList([ torch.nn.parameter.Parameter(mean[i], requires_grad=True) for i in range(mean.shape[0]) ])
        self.covarmanifold : geoopt.SymmetricPositiveDefinite = geoopt.SymmetricPositiveDefinite()
        self.covariance : torch.nn.ParameterList = torch.nn.ParameterList([ geoopt.ManifoldParameter(self.covarmanifold.projx(covariance[i]), requires_grad=True, manifold=self.covarmanifold) for i in range(covariance.shape[0]) ])
    def forward(self, M : torch.Tensor):
        mean : torch.Tensor = torch.stack([p for p in self.mean], dim=0)
        covariance : torch.Tensor = torch.stack([p for p in self.covariance], dim=0)
        pointsout : torch.Tensor = torch.matmul(M, mean)
        msquare : torch.Tensor = torch.square(M)
        pointscovarout : torch.Tensor = torch.sum(msquare[:,:,None,None]*covariance, dim=1)
        return pointsout, pointscovarout