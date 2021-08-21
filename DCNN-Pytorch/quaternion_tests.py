import numpy as np
import torch
from scipy.spatial.transform.rotation import Rotation as Rot
import deepracing_models.math_utils.quaternion as quat_utils


device = torch.device("cpu")
# device = torch.device("cuda:0")
Nrots = 1
vecs : torch.Tensor = torch.randn(Nrots, 4, dtype=torch.float64, device=device)
norms = torch.norm(vecs, p=2, dim=1)
quat = vecs/norms[:,None]
rotsp = Rot.from_quat(quat.clone().detach().cpu().numpy())


rmattorch = quat_utils.quaternionToMatrix(quat)
rmatfromsp = torch.as_tensor(rotsp.as_matrix(), dtype=rmattorch.dtype, device=rmattorch.device)
print(rmattorch - rmatfromsp)
print(torch.allclose(rmattorch, rmatfromsp))


rmattorch0 = quat_utils.quaternionToMatrix(quat[0])
rmatfromsp0 = torch.as_tensor(rotsp.as_matrix()[0], dtype=rmattorch0.dtype, device=rmattorch0.device)
print(rmattorch0 - rmatfromsp0)
print(torch.allclose(rmattorch0, rmatfromsp0))


