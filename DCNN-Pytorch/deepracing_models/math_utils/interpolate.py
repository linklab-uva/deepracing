import torch
import torch.nn, torch.nn.parameter
import io
class LinearInterpolator(torch.nn.Module):
    def __init__(self, x_points : torch.Tensor, y_points : torch.Tensor, requires_grad=False):
        super(LinearInterpolator,self).__init__()
        self.x_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=x_points, requires_grad=requires_grad)
        self.dx : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=x_points[1:] - x_points[:-1], requires_grad=requires_grad)
        self.y_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=y_points, requires_grad=requires_grad)
        self.dy : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=y_points[1:] - y_points[:-1], requires_grad=requires_grad)
    def forward(self, x_samp : torch.Tensor):
        x_samp_flat = x_samp.view(-1)
        idxbuckets = (torch.bucketize(x_samp_flat, self.x_points, right=True) - 1)#.clamp(min=0, max=self.x_points.shape[0]-1)
        idxbuckets_negative = torch.any(idxbuckets<0)
        idxbuckets_toobig = torch.any(idxbuckets>=self.dx.shape[0])
        if idxbuckets_negative or idxbuckets_toobig:
            strio = io.StringIO()
            print("idxbuckets must be nonnegative" if idxbuckets_negative else "idxbuckets must be less than self.dx.shape[0] (%d)" % (self.dx.shape[0],), file=strio)
            print("self.x_points:", self.x_points, file=strio)
            print("self.dx:", self.dx, file=strio)
            print("x_samp:", x_samp, file=strio)
            print("x_samp_flat:", x_samp_flat, file=strio)
            print("idxbuckets:", idxbuckets, file=strio)
            strio.flush()
            raise ValueError(strio.getvalue()) 
        xi = self.x_points[idxbuckets]
        dx = self.dx[idxbuckets]
        yi = self.y_points[idxbuckets]
        dy = self.dy[idxbuckets]
        ds = (x_samp_flat-xi)/dx
        if self.y_points.ndim>1:
            return (yi+ds[:,None]*dy).view(*x_samp.shape, self.y_points.shape[-1])
        else:
            return (yi+ds*dy).view(x_samp.shape)


