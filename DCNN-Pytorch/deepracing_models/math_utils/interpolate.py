import torch
import torch.nn, torch.nn.parameter

class LinearInterpolator(torch.nn.Module):
    def __init__(self, x_points : torch.Tensor, y_points : torch.Tensor, requires_grad=False):
        super(LinearInterpolator,self).__init__()
        self.x_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=x_points, requires_grad=requires_grad)
        self.dx : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=x_points[1:] - x_points[:-1], requires_grad=requires_grad)
        self.y_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=y_points, requires_grad=requires_grad)
        self.dy : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=y_points[1:] - y_points[:-1], requires_grad=requires_grad)
    def forward(self, x_samp : torch.Tensor):
        x_samp_flat = x_samp.view(-1)
        idxbuckets = torch.bucketize(x_samp_flat, self.x_points, right=True) - 1
        xi = self.x_points[idxbuckets]
        dx = self.dx[idxbuckets]
        yi = self.y_points[idxbuckets]
        dy = self.dy[idxbuckets]
        ds = (x_samp_flat-xi)/dx
        return (yi+ds*dy).view(x_samp.shape)


