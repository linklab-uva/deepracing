import collections, collections.abc
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.figure, matplotlib.axes
import torch.utils.data as torchdata
from texttable import Texttable
import latextable
import torch
import yaml
title_dict : dict = {
    "ade" : "MinADE",
    "fde" : "FDE",
    "lateral_error" : "Lateral Error",
    "longitudinal_error" : "Longitudinal Error"
}
class PredictionResults(collections.abc.Mapping[str,np.ndarray]):
    def __init__(self, resultsdict : dict[str,np.ndarray], data_dir : str, modelname : str) -> None:
        self.resultsdict = resultsdict
        self.data_dir = data_dir
        self.modelname = modelname
    def __eq__(self, __other: 'PredictionResults') -> bool:
        return self.modelname == __other.modelname
    def __ne__(self, __other: 'PredictionResults') -> bool:
        return not (self.modelname == __other.modelname)
    def __hash__(self):
        return hash(self.modelname)
    def __iter__(self):
        return iter(self.resultsdict)
    def __len__(self):
        return len(self.resultsdict)
    def __getitem__(self, key):
        return self.resultsdict[key]
    def __setitem__(self, key, value):
        self.resultsdict[key] = value
    def numsamples(self):
        return self.resultsdict["predictions"].shape[0]
    def keys(self):
        return self.resultsdict.keys()
    def error_summary(self, p0=2, pf=98) -> dict[str, dict[str, float | int]]:
        rtn : dict = {}
        for k in ["ade", "fde", "longitudinal_error", "lateral_error"]:
            err = self.resultsdict[k]
            minval = float(np.percentile(err, p0))
            maxval = float(np.percentile(err, pf))
            num_outliers = int(np.sum(err>maxval))
            rtn[k] = {
                "mean" : float(np.mean(err)),
                "min" : float(np.min(err)),
                "max" : float(np.max(err)),
                "stdev" : float(np.std(err)),
                "num_outliers" : num_outliers,
                "percentile_%d" % (p0,) : minval,
                "percentile_%d" % (pf,) : maxval
            }
        return rtn
    def subsample(self, idx : np.ndarray, copy=False):
        if copy:
            resultsdict = {
                k : v[idx].copy() for (k,v) in self.resultsdict.items()
            }
        else:
            resultsdict = {
                k : v[idx] for (k,v) in self.resultsdict.items()
            }
            
        return PredictionResults(resultsdict, self.data_dir, self.modelname)
    def trim_iqr(self, whis : float = 1.5, metric : str = "ade"):
        err = self.resultsdict[metric]
        p0_value = np.percentile(err, 25)
        pf_value = np.percentile(err, 75)
        iqr = pf_value - p0_value
        maxval = p0_value + whis*iqr
        return err<=maxval, maxval
    def trim_percentiles(self, pf : float = 95.0, metric : str = "ade"):
        err = self.resultsdict[metric]
        maxval = np.percentile(err, pf)
        return err<=maxval, maxval
    

    @staticmethod
    def from_data_file(data_file : str, modelname : str, sort_idx : np.ndarray | None = None, allow_pickle=False) -> 'PredictionResults':
        data_dir = os.path.dirname(data_file)
        with open(data_file, "rb") as f:
            results_file = np.load(f, allow_pickle=allow_pickle)
            if sort_idx is None:
                results_dict = {k: v.copy() for (k,v) in results_file.items()}
            else:
                results_dict = {k: v[sort_idx].copy() for (k,v) in results_file.items()}
        return PredictionResults(results_dict, data_dir, modelname)
    def compute_fde(self):
        self.resultsdict["fde"] = np.linalg.norm(self.resultsdict["predictions"][:,-1,[0,1]] - self.resultsdict["ground_truth"][:,-1,[0,1]], ord=2.0, axis=1)

def plot_error_histograms(results_list : list[PredictionResults], plotbase : str, 
                          metric="ade", bins=200, notch=True, pad_inches=0.02, 
                          whis=2.0, combined_subdir="combined"):
    title = title_dict[metric]
    key = metric
    # fig_combined_histogram, axes_list_histogram = plt.subplots(1, len(results_list)) 
    fig_combined_histogram, _axes_histogram_ = plt.subplots() 
    axes_histogram : matplotlib.axes.Axes = _axes_histogram_
    # fig_combined_histogram.suptitle(title)
    fig_combined_boxplot, axes_list_boxplot = plt.subplots(1, len(results_list)) 
    # fig_combined_boxplot.suptitle(title)
    max_error = float(np.max([float(np.max(results[key])) for results in results_list]))
    for (i, results) in enumerate(results_list):
        # axes_histogram : matplotlib.axes.Axes = axes_list_histogram[i]
        errors = results[key]
        modelname = results.modelname
        axes_histogram.hist(errors, bins=bins, label=modelname)
        # axes_histogram.set_ylim(bottom=0, top=1.01*max_error)
        # axes_histogram.set_title(modelname)


        axes_boxplot : matplotlib.axes.Axes = axes_list_boxplot[i]
        axes_boxplot.set_title(modelname, x=0.8, fontsize=11, y=0.5)
        axes_boxplot.boxplot(errors, notch=notch, whis=whis)
        axes_boxplot.set_ylim(bottom=0, top=1.01*max_error)
    fig_combined_histogram.legend()
    savedir = os.path.join(plotbase, combined_subdir)
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)
    fig_combined_histogram.tight_layout(pad=0.1)
    # fig_combined_histogram.savefig(os.path.join(savedir, "histogram.png"), backend="agg", pad_inches=pad_inches)
    fig_combined_histogram.savefig(os.path.join(savedir, "histogram.pdf"), backend="pdf", pad_inches=pad_inches)
    fig_combined_histogram.savefig(os.path.join(savedir, "histogram.svg"), backend="svg", pad_inches=pad_inches)
    plt.close(fig=fig_combined_histogram)
    fig_combined_boxplot.tight_layout(pad=0.1)
    # fig_combined_boxplot.savefig(os.path.join(savedir, "boxplot.png"), backend="agg", pad_inches=pad_inches)
    fig_combined_boxplot.savefig(os.path.join(savedir, "boxplot.pdf"), backend="pdf", pad_inches=pad_inches)
    fig_combined_boxplot.savefig(os.path.join(savedir, "boxplot.svg"), backend="svg", pad_inches=pad_inches)
    plt.close(fig=fig_combined_boxplot)

    for results in results_list:
        savedir = os.path.join(plotbase, results.modelname)
        if os.path.isdir(savedir):
            shutil.rmtree(savedir)
        os.makedirs(savedir)
        errors = results[key]
        modelname = results.modelname
        
        fig : matplotlib.figure.Figure = plt.figure()
        plt.hist(errors, bins=bins)
        plt.title(title + ": " + modelname)
        fig.savefig(os.path.join(savedir, "histogram.png"), backend="agg")
        fig.savefig(os.path.join(savedir, "histogram.pdf"), backend="pdf")
        fig.savefig(os.path.join(savedir, "histogram.svg"), backend="svg")
        plt.close(fig=fig)

        figbox : matplotlib.figure.Figure = plt.figure()
        plt.title(title + ": " + modelname)
        plt.boxplot(errors, notch=notch, whis=whis)
        figbox.savefig(os.path.join(savedir, "boxplot.png"), backend="agg")
        figbox.savefig(os.path.join(savedir, "boxplot.pdf"), backend="pdf")
        figbox.savefig(os.path.join(savedir, "boxplot.svg"), backend="svg")
        plt.close(fig=figbox)
        
def plot_outliers(results_list : list[PredictionResults], plotdir : str, fulldset : torchdata.Dataset, 
                  metric="ade", N=1, worst=True, with_history=True, ref_alpha=1.0, nonref_alpha=0.25):
    if plotdir is None or (not type(plotdir)==str):
        raise ValueError("plotdir must be a string")
    if os.path.isfile(plotdir):
        raise ValueError("plotdir must be a directory")
    ref_results = results_list[0]
    Nclipped = min(N, ref_results[metric].shape[0] - 1)
    idx_sort = np.argsort(ref_results[metric])
    if worst:
        idx_sort = np.flipud(idx_sort)
        subdir_name="bottom_%d_%s_%s" % (Nclipped, ref_results.modelname, metric)
    else:
        subdir_name="top_%d_%s_%s" % (Nclipped, ref_results.modelname, metric)
    plotdirfull = os.path.join(plotdir, subdir_name)
    if os.path.isdir(plotdirfull):
        shutil.rmtree(plotdirfull)
    os.makedirs(plotdirfull)
    for plot_idx in range(Nclipped):
        dset_dict : dict[str, np.ndarray] = fulldset[idx_sort[plot_idx]]
        history = dset_dict["hist"][40:]
        ground_truth = dset_dict["fut"]
        history_vel = dset_dict["hist_vel"]
        # print(history_vel.T)
        left_bound = ref_results["left_bd"][idx_sort[plot_idx]]
        right_bd = ref_results["right_bd"][idx_sort[plot_idx]]
        history_speed = np.linalg.norm(history_vel, ord=2.0, axis=1)
        thistory = dset_dict["thistory"]
        ground_truth_vel = dset_dict["fut_vel"]
        ground_truth_speed = np.linalg.norm(ground_truth_vel, ord=2.0, axis=1)
        tfuture = dset_dict["tfuture"]
        fig : matplotlib.figure.Figure = plt.figure()
        if with_history:
            plt.plot(-history[:,1], history[:,0], label="History", linestyle="--", c="grey")
        plt.scatter(-ground_truth[:,1], ground_truth[:,0], label="Ground Truth", c="grey", alpha=0.5, s=10.0)
        predictions = ref_results["predictions"][idx_sort[plot_idx]]
        plt.plot(-predictions[:,1], predictions[:,0], label=ref_results.modelname, alpha=ref_alpha)
        for (idx, results) in enumerate(results_list):
            if results==ref_results:
                continue
            predictions = results["predictions"][idx_sort[plot_idx]]
            plt.plot(-predictions[:,1], predictions[:,0], label=results.modelname, alpha=nonref_alpha)
        plt.legend()
        plt.plot(-left_bound[:,1], left_bound[:,0], c="black")
        plt.plot(-right_bd[:,1], right_bd[:,0], c="black")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.tight_layout()
        # fig.savefig(os.path.join(plotdirfull, "sample_%d.svg" % (plot_idx,)))
        fig.savefig(os.path.join(plotdirfull, "sample_%d.pdf" % (plot_idx,)), backend="pdf")
        fig.savefig(os.path.join(plotdirfull, "sample_%d.png" % (plot_idx,)), backend="agg")
        # fig.savefig(os.path.join(plotdirfull, "sample_%d.pgf" % (plot_idx,)), backend="pgf")
        plt.close(fig=fig)
        fig_speed = plt.figure()
        plt.plot(thistory, history_speed, label="History", linestyle="--", c="grey")
        plt.plot(tfuture, ground_truth_speed, label="Ground Truth", c="grey")
        for (idx, results) in enumerate(results_list):
            if "vel_predictions" in results.keys():
                plt.plot(tfuture, np.linalg.norm(results["vel_predictions"][idx_sort[plot_idx]], ord=2.0, axis=1), label=results.modelname)
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speed (m/s)")
        plt.tight_layout()
        # fig_speed.savefig(os.path.join(plotdirfull, "sample_%d_speed.svg" % (plot_idx,)))
        fig_speed.savefig(os.path.join(plotdirfull, "sample_%d_speed.pdf" % (plot_idx,)), backend="pdf")
        fig_speed.savefig(os.path.join(plotdirfull, "sample_%d_speed.png" % (plot_idx,)), backend="agg")
        # fig_speed.savefig(os.path.join(plotdirfull, "sample_%d_speed.pgf" % (plot_idx,)), backend="pgf")
        plt.close(fig=fig_speed)
    return idx_sort
def create_table(results : list[PredictionResults]) -> Texttable:
    def boldstring(input_string : str):
        return input_string
        # return color.BOLD + input_string + color.END
    texttable = Texttable(max_width=0)
    title_to_key = {
        "ADE" : "ade",
        "Lateral\nError" : "lateral_error",
        "Longitudinal\nError" : "longitudinal_error",
        "FDE" : "fde"
    }
    column_names = ["Model"] + sorted(title_to_key.keys())
    texttable.set_cols_align(["c"]*len(column_names))
    texttable.set_cols_valign(["m"]*len(column_names))
    texttable.header([boldstring(s) for s in column_names])
    # texttable.add_row([boldstring(s) for s in column_names])
    for result in results:
        texttable.add_row([result.modelname] + [str(np.mean(result[title_to_key[cname]])) for cname in column_names[1:]])
    return texttable
#metric_key="ade", N=1, worst=True, with_history=True, ref_alpha=1.0, nonref_alpha=0.25
def cross_error_analysis(results_list : list[PredictionResults], 
                         fulldset : torchdata.Dataset, 
                         basedir : str, 
                        **kwargs):
    argdict = {
        "metric" : "ade",
        "N" : 10,
        "histograms" : True,
        "with_history" : True,
        "ref_alpha" : 1.0,
        "nonref_alpha" : 0.25,
        "pf": None,
        "whis": None,
        "idx_filter" : None,
        "subdir" : None,
        "bins" : 100,
        "notch" : True,
        "other_models" : []
    }
    argdict.update(kwargs)
    whis : float | None = argdict["whis"]
    pf : float | None = argdict["pf"]
    idx_filter : np.ndarray | None = argdict["idx_filter"]
    metric=argdict["metric"]
    if len([x for x in [whis, pf, idx_filter] if x is not None ])>1:
        raise ValueError("Must specify no more than 1 of 'whis', 'pf', or 'idx_filter'")
    if pf is not None:
        subdir = os.path.join(basedir, "trim_%s_%s_%d_percentile" % (results_list[0].modelname, argdict["metric"], pf))
        idxgood, _ = results_list[0].trim_percentiles(**{k : argdict[k] for k in ["pf", "metric"]})
        whis = (0, pf)
    elif whis is not None:
        subdir = os.path.join(basedir, "trim_%s_%s_%3.3f_iqr" % (results_list[0].modelname, argdict["metric"], whis))
        idxgood, _ = results_list[0].trim_iqr(**{k : argdict[k] for k in ["whis", "metric"]})
    elif idx_filter is not None:
        subdir_arg : str | None = argdict["subdir"]
        if subdir_arg is None:
            raise ValueError("'subdir' must also be specified if 'idx_filter' is specified")
        subdir = os.path.normpath(os.path.join(basedir, subdir_arg))
        idxgood = idx_filter
        whis = (0, 98)
    else:
        subdir = os.path.join(basedir, "baseline")
        idxgood = np.ones(results_list[0][metric].shape[0], dtype=bool)
        whis = (0, 98)
    results_trimmed_list : list[PredictionResults] = [r.subsample(idxgood) for r in results_list]
    dset_trimmed : torchdata.Subset = torchdata.Subset(fulldset, np.where(idxgood)[0])
    if argdict["histograms"]:
        histogramdir = os.path.join(subdir, "histograms")
        plot_error_histograms(results_trimmed_list, histogramdir, whis = whis, metric=metric, bins=argdict["bins"], notch=argdict["notch"])
    plotdir = os.path.join(subdir, "plots")
    plot_outliers(results_trimmed_list, plotdir, dset_trimmed, 
                  N=argdict["N"], worst=True, 
                  with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                  nonref_alpha=argdict["nonref_alpha"])
    plot_outliers(results_trimmed_list, plotdir, dset_trimmed, 
                  N=argdict["N"], worst=False, 
                  with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                  nonref_alpha=argdict["nonref_alpha"])
    for results_trimmed in results_trimmed_list:
        with open(os.path.join(subdir, "%s_summary.yaml" % (results_trimmed.modelname)), "w") as f:
            yaml.safe_dump(results_trimmed.error_summary(), f, indent=2)
    result_set = set(results_trimmed_list)
    result_dict = {res.modelname : res for res in results_trimmed_list}
    other_models : list[str] = list(set(argdict["other_models"]))
    for model_name in other_models:
        result_to_plot = result_dict[model_name]
        other_results = result_set - {result_to_plot,}
        plot_outliers([result_to_plot,] + list(other_results), plotdir, dset_trimmed, 
                    N=argdict["N"], worst=False, 
                    with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                    nonref_alpha=argdict["nonref_alpha"])
        plot_outliers([result_to_plot,] + list(other_results), plotdir, dset_trimmed, 
                    N=argdict["N"], worst=True, 
                    with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                    nonref_alpha=argdict["nonref_alpha"])

        




        

