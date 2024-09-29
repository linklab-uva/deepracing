import torch
import deepracing_models.math_utils.polynomial as polyutils
def parameterize_time(speeds : torch.Tensor, arclengths : torch.Tensor):
    times = torch.zeros_like(speeds)
    for i in range(1, times.shape[0]-1):
        dr = arclengths[i] - arclengths[i-1]
        v0 = speeds[i-1]
        vf = speeds[i]
        accel = (vf**2 - v0**2)/(2.0*dr)
        if accel.abs()<1E-5:
            times[i] = times[i-1] + dr/v0
        else:
            coefs = torch.stack([-dr, v0, 0.5*accel])
            roots : torch.Tensor = polyutils.polyroots(coefs[None])[0].real
            positiveroots = roots[roots>0.0]
            times[i] = times[i-1] + positiveroots.min()
    times[-1] = times[-2] + (arclengths[-1] - arclengths[-2])/speeds[-2]
    return times