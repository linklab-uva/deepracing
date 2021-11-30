import torch
import torch.nn, torch.nn.parameter

class LinearInterpolator(torch.nn.Module):
    def __init__(self, x_points : torch.Tensor, y_points : torch.Tensor, requires_grad=False):
        super(LinearInterpolator,self).__init__()
        self.x_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=x_points, requires_grad=requires_grad)
        self.y_points : torch.nn.parameter.Parameter = torch.nn.parameter.Parameter(data=y_points, requires_grad=requires_grad)
    def forward(self, x_samp : torch.Tensor):
        batch = x_samp.shape[0]
        x_points : torch.Tensor = self.x_points.unsqueeze(0).expand(batch, -1)
        y_points : torch.Tensor = self.y_points.unsqueeze(0).expand(batch, -1)
        xdeltas = (x_samp.unsqueeze(2)-x_points[:,None]).squeeze(2)
        xdeltas[xdeltas<0]=float("inf")
        idxi = torch.argmin(xdeltas, dim=2, keepdim=False)
        idxf = idxi + 1
        speedsi=torch.stack([x_points[i,idxi[i]] for i in range(batch)], dim=0)
        speedsf=torch.stack([x_points[i,idxf[i]] for i in range(batch)], dim=0)
        limitsi=torch.stack([y_points[i,idxi[i]] for i in range(batch)], dim=0)
        limitsf=torch.stack([y_points[i,idxf[i]] for i in range(batch)], dim=0)
        dx = speedsf-speedsi
        dy = limitsf-limitsi
        ds = (x_samp-speedsi)/dx
        return limitsi+ds*dy


