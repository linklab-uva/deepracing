import torch
import torch.nn
import typing


class TrackPaths(torch.nn.Module):
    def __init__(self, innerbound : torch.Tensor, outerbound : torch.Tensor, raceline : torch.Tensor, centerline : typing.Union[torch.Tensor,None] = None ) -> None:
        super(TrackPaths, self).__init__()
        self.innerbound : torch.nn.Parameter = torch.nn.Parameter(innerbound.clone(), requires_grad=False)
        self.outerbound : torch.nn.Parameter = torch.nn.Parameter(outerbound.clone(), requires_grad=False)
        self.raceline : torch.nn.Parameter = torch.nn.Parameter(raceline.clone(), requires_grad=False)
        if centerline is None:
            self.centerline : torch.nn.Parameter = torch.nn.Parameter((0.5*(innerbound + outerbound)).clone(), requires_grad=False)
        else:
            self.centerline : torch.nn.Parameter = torch.nn.Parameter(centerline.clone(), requires_grad=False)
    def forward(self, x : torch.Tensor):
        #This module doesn't actually do anything. it's just a convenient container for the 4 reference paths commonly used in DeepRacing.
        return x