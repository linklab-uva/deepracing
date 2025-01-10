import torch

def rectangle_intersections(box1 : torch.Tensor, box2 : torch.Tensor, combinations = torch.as_tensor([  [0, 1],
                                                                                                        [0, 2],
                                                                                                        [0, 3],
                                                                                                        [1, 0],
                                                                                                        [1, 2],
                                                                                                        [1, 3],
                                                                                                        [2, 0],
                                                                                                        [2, 1],
                                                                                                        [2, 3],
                                                                                                        [3, 0],
                                                                                                        [3, 1],
                                                                                                        [3, 2]]  )):
    
    lhs = torch.stack([
        box1[:, (combinations[:,0] + 1)%4] - box1[:, combinations[:,0]],
        box2[:, combinations[:,1]] - box2[:, (combinations[:,1] + 1)%4],
        ], dim=-1)
    rhs = box2[:,combinations[:,1]] - box1[:,combinations[:,0]]
    solution : torch.Tensor = (torch.linalg.solve(lhs, rhs))#[0]
    intersection_idx = ((solution[:,:,0]>=0.0)*(solution[:,:,0]<=1.0)*(solution[:,:,1]>=0.0)*(solution[:,:,1]<=1.0)).cpu()
    collision_idx = torch.sum(intersection_idx, dim=1)>0
    return solution, intersection_idx, collision_idx, combinations