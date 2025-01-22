import torch
import collections
RectangleIntersections = collections.namedtuple('RectangleIntersections', ['solution', 'intersection_idx', 'collision_idx', 'combinations'])
def rectangle_intersections1(box1 : torch.Tensor, box2 : torch.Tensor):
    tau1 = box1[...,1,:] - box1[...,0,:]
    box1lengths = torch.norm(tau1, p=2.0, dim=-1, keepdim=False)
    box1widths = torch.norm(box1[...,2,:] - box1[...,1,:], p=2.0, dim=-1, keepdim=False)
    # print(box1lengths, box1widths)
    tau1 = tau1/(box1lengths[...,None])
    R1 = torch.stack([tau1, tau1[...,[1,0]]], dim=-2)
    R1[...,1,0]*=-1.0
    p1 = -(R1@torch.mean(box1, dim=-2)[...,None])[...,0]

    tau2 = box2[...,1,:] - box2[...,0,:]
    box2lengths = torch.norm(tau2, p=2.0, dim=-1, keepdim=False)
    box2widths = torch.norm(box2[...,2,:] - box2[...,1,:], p=2.0, dim=-1, keepdim=False)
    # print(box2lengths, box2widths)
    tau2 = tau2/(box2lengths[...,None])
    R2 = torch.stack([tau2, tau2[...,[1,0]]], dim=-2)
    R2[...,1,0]*=-1.0
    p2 = -(R2@torch.mean(box2, dim=-2)[...,None])[...,0]

    p2_in_C1 = (R1[...,None,:,:]@box2[...,None])[...,0] + p1[...,None,:]
    p1_in_C2 = (R2[...,None,:,:]@box1[...,None])[...,0] + p2[...,None,:]
   
    box1lengths_exp = box1lengths[...,None].expand_as(p2_in_C1[...,0])
    box2lengths_exp = box2lengths[...,None].expand_as(box1lengths_exp)

    box1widths_exp = box1widths[...,None].expand_as(box1lengths_exp)
    box2widths_exp = box2widths[...,None].expand_as(box1lengths_exp)

    collision_idx = (torch.sum( ((p2_in_C1[...,0]>=-box1lengths_exp*0.5) * (p2_in_C1[...,0]<=box1lengths_exp*0.5))*((p2_in_C1[...,1]>=-box1widths_exp*0.5) * (p2_in_C1[...,1]<=box1widths_exp*0.5)), dim=-1)>0) +\
                    (torch.sum( ((p1_in_C2[...,0]>=-box2lengths_exp*0.5) * (p1_in_C2[...,0]<=box2lengths_exp*0.5))*((p1_in_C2[...,1]>=-box2widths_exp*0.5) * (p1_in_C2[...,1]<=box2widths_exp*0.5)), dim=-1)>0) 
    return collision_idx
def rectangle_intersections2(box1 : torch.Tensor, box2 : torch.Tensor, check_singular : bool = True, combinations = torch.as_tensor([  [0, 1],
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
        box1[..., (combinations[:,0] + 1)%4, :] - box1[..., combinations[:,0], :],
        box2[..., combinations[:,1], :] - box2[..., (combinations[:,1] + 1)%4, :],
        ], dim=-1)
    # print(lhs[0])
    rhs = box2[..., combinations[:,1], :] - box1[..., combinations[:,0], :]
    solution = torch.nan*torch.empty_like(rhs)
    #(lhs[...,0,0]*lhs[...,1,1] - lhs[...,1,0]*lhs[...,0,1])
    valid_idx = torch.abs(torch.linalg.det(lhs))>1E-5 if check_singular else torch.ones(rhs.shape[0], dtype=bool)
    solution[valid_idx] = (torch.linalg.solve(lhs[valid_idx], rhs[valid_idx]))#[0]
    intersection_idx = ((solution[...,0]>=0.0)*(solution[...,0]<=1.0)*(solution[...,1]>=0.0)*(solution[...,1]<=1.0))
    # print(solution.device)
    # print(intersection_idx.device)
    collision_idx = torch.sum(intersection_idx, dim=-1)>0
    # print(collision_idx.device)
    return RectangleIntersections(solution, intersection_idx, collision_idx, combinations)
# def rectangle_intersections(box1 : torch.Tensor, box2 : torch.Tensor, check_singular : bool = True, combinations = torch.as_tensor([
#                                                                                                         # [0, 0],
#                                                                                                         # [0, 2],
#                                                                                                         [2, 2],
#                                                                                                         [2, 0],])):
    
#     result = rectangle_intersections2(box1, box2, check_singular=check_singular)
#     # broadphase_idx = rectangle_intersections1(box1, box2)
#     # maybe = ~broadphase_idx
#     # # print(box1.device)
#     # # print(box2.device)
#     # # print(maybe.device)
#     # result = rectangle_intersections2(box1[maybe], box2[maybe], check_singular=check_singular, combinations=combinations)
#     # # print(broadphase_idx.device)
#     # # print(result.collision_idx.device)
#     # broadphase_idx[maybe.to(device=broadphase_idx.device)] = result.collision_idx
#     return result.collision_idx, result