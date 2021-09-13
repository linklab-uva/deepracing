import argparse
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np, torch
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Test AdmiralNet")
    parser.add_argument("steer", type=float)

    boundsy = torch.linspace(0.0,20.0, steps=100, dtype = torch.float64)
    leftbound = torch.stack([-3.0*torch.ones_like(boundsy), boundsy, torch.zeros_like(boundsy)], dim=1)
    rightbound = torch.stack([3.0*torch.ones_like(boundsy), boundsy, torch.zeros_like(boundsy)], dim=1)
    raceline = torch.stack([-1.5*torch.ones_like(boundsy), boundsy, torch.zeros_like(boundsy)], dim=1)
    #R = torch.eye(3, device = boundsy.device, dtype = boundsy.dtype )
    R = torch.as_tensor( Rotation.from_rotvec(0.0*np.array([0.0,0.0,1.0])).as_matrix() , device = boundsy.device, dtype = boundsy.dtype)
    
    obstacle = torch.as_tensor([[-2, 10, 0],\
                                [-2, 11, 0],\
                                [-1, 10, 0],\
                                [-1, 11, 0], ], dtype=boundsy.dtype, device=boundsy.device)
    obstacle_centroid = torch.mean(obstacle,dim=0)
    #T[0:3,3] = torch.mean(obstacle[:,0:3],dim=0)
    #T = torch.inverse(T)
    obstacle = torch.matmul(obstacle-obstacle_centroid[None,:], R.t())+obstacle_centroid[None,:]
    print(obstacle_centroid)
    print(obstacle)
    obstacle2 = torch.as_tensor([[2, 5, 0],\
                                [2, 6, 0],\
                                [1, 5, 0],\
                                [1, 6, 0], ], dtype=boundsy.dtype, device=boundsy.device)


    ego = torch.as_tensor([[-0.5, -2.5, 0],\
                           [-0.5, 2.5, 0],\
                           [0.5, -2.5, 0],\
                           [0.5, 2.5, 0],
                            ], dtype=boundsy.dtype, device=boundsy.device) 
    pntsets = [leftbound, rightbound]
    pntsets.append(obstacle)
    pntsets.append(obstacle2)
  #  pntsets.append(raceline)
  #  pntsets.append(ego)
    pts = torch.cat(pntsets, dim=0)
    vor = Voronoi(pts[:,0:2].cpu().numpy(), furthest_site=False)
    voronoi_plot_2d(vor)
    plt.show()
    
if __name__ == '__main__':
    main()