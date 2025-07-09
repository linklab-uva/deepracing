from genericpath import isdir
import torch
import yaml
import numpy as np
import os
import shutil
def save_results(outputdir : str, predictions : torch.Tensor, ground_truths : torch.Tensor, computation_time : torch.Tensor):

    errorvals = (predictions - ground_truths)
    abserrorvals = errorvals.abs()
    overestimating_idx = (predictions>=ground_truths)
    underestimating_idx = (predictions<ground_truths)
    overestimating_percentage = (overestimating_idx.sum()/overestimating_idx.shape[0])
    underestimating_percentage = (underestimating_idx.sum()/underestimating_idx.shape[0])
    comptimes = computation_time[1:]
    summary = {
        "comptime_mean" : comptimes[1:].mean().item(),
        "comptime_median" : comptimes[1:].median().item(),
        "looprate_mean" : (1.0/comptimes[1:]).mean().item(),
        "looprate_median" : (1.0/comptimes[1:]).median().item(),
        "MAE" : abserrorvals.mean().item(),
        "MAE_stdev" : abserrorvals.std().item(),
        "MAE_over" : abserrorvals[overestimating_idx].mean().item(),
        "MAE_under" : abserrorvals[underestimating_idx].mean().item(),
        "Overestimating Percentage" : overestimating_percentage.item(),
        "Underestimating Percentage" : underestimating_percentage.item(),
    }
    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)
    with open(os.path.join(outputdir, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f)
    with open(os.path.join(outputdir, "results.npz"), "wb") as f:
        np.savez(f, predictions=predictions.cpu().numpy(),
                    ground_truths=ground_truths.cpu().numpy(),
                    comp_times = computation_time.cpu().numpy())
    return summary
