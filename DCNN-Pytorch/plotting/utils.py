import collections, collections.abc
from ftplib import all_errors
from math import e
from typing import Iterable
import numpy as np
import os
import shutil
import matplotlib.transforms
import matplotlib.pyplot as plt
import matplotlib.figure, matplotlib.axes
import torch.utils.data as torchdata
from texttable import Texttable
import latextable
import torch
import yaml
import seaborn as sns
title_dict : dict = {
    "ade" : "MinADE",
    "fde" : "FDE",
    "lateral_error" : "Lateral Error",
    "longitudinal_error" : "Longitudinal Error"
}
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
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
                "median" : float(np.median(err)),
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
        maxval = float(np.percentile(err, pf))
        return err<=maxval, maxval
    

    @staticmethod
    def from_data_file(data_file : str, modelname : str, sort_idx : np.ndarray | None = None, allow_pickle=False) -> 'PredictionResults':
        print("Loading results for model %s from %s" % (modelname, data_file))
        data_dir = os.path.dirname(data_file)
        with open(data_file, "rb") as f:
            results_file = np.load(f, allow_pickle=allow_pickle)
            if sort_idx is None:
                results_dict = {k: v.copy() for (k,v) in results_file.items()}
            else:
                results_dict = {k: v[sort_idx].copy() for (k,v) in results_file.items()}
        print("Done")
        return PredictionResults(results_dict, data_dir, modelname)
    def compute_fde(self):
        self.resultsdict["fde"] = np.linalg.norm(self.resultsdict["predictions"][:,-1,[0,1]] - self.resultsdict["ground_truth"][:,-1,[0,1]], ord=2.0, axis=1)
import scipy.interpolate
import scipy.stats
class CustomScaleHelper():
    def __init__(self, tickvals : np.ndarray):
        self.lin = np.linspace(0.0, 1.0, num=tickvals.shape[0])
        self.tickvals = np.sort(tickvals)
        self.spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(self.tickvals, self.lin, k=1)
        self.inv_spline : scipy.interpolate.BSpline = scipy.interpolate.make_interp_spline(self.lin, self.tickvals, k=1)
    def forward(self, val : float | np.ndarray):
        return self.spline(val)
    def inverse(self, val : float | np.ndarray):
        return self.inv_spline(val)
import matplotlib.scale
import matplotlib.container
import matplotlib.patches
import matplotlib.collections
from matplotlib.gridspec import GridSpec
# "png"
def plot_individually(results_list : list[PredictionResults], **kwargs):
    vert = kwargs["vert"]
    box_plot_maxes = kwargs["box_plot_maxes"]
    showfliers = kwargs["showfliers"]
    key = kwargs["key"]
    whis = kwargs["whis"]
    vertlines = kwargs["vertlines"]
    bins = kwargs["bins"]
    box_plot_scale = kwargs["box_plot_scale"]
    notch = kwargs["notch"]
    scale_ticks = kwargs["scale_ticks"]
    if vert:
        nrows = 1
        ncols = len(results_list)
    else:
        nrows = len(results_list)
        ncols = 1
    # fig_combined_violinplot, _axes_list_violinplot_ = plt.subplots(nrows=nrows, ncols=ncols) 
    fig_combined_violinplot = plt.figure(layout="constrained")
    axes_list_violinplot : list[matplotlib.axes.Axes] = []
    grid_size = 100
    split_ratio = 1.0 # 0.65
    split_integer = int(round(split_ratio*grid_size))
    gs = GridSpec(nrows, grid_size, figure=fig_combined_violinplot)
    for row in range(nrows-1):
        axes_list_violinplot.append(fig_combined_violinplot.add_subplot(gs[row, 0:split_integer]))
    axes_list_violinplot.append(fig_combined_violinplot.add_subplot(gs[-1, :]))

    fig_combined_boxplot, _axes_list_boxplot_ = plt.subplots(nrows=nrows, ncols=ncols) 
    axes_list_boxplot : list[matplotlib.axes.Axes] = _axes_list_boxplot_
    fig_combined_histogram, _axes_histogram_ = plt.subplots() 
    axes_histogram : matplotlib.axes.Axes = _axes_histogram_

    if box_plot_maxes is None:
        box_plot_maxes = {res.modelname : 1.05*float(np.max(res[key])) if showfliers else 1.05*float(np.percentile(res[key], whis[1])) for res in results_list}
    elif type(box_plot_maxes) is float:
        c = float(box_plot_maxes)
        box_plot_maxes = {res.modelname : (c*(split_integer/grid_size) if res!=results_list[-1] else c) for res in results_list}
    count_maxes = []
    vline_vals = []
    vline_colors = []
    binmedians_list = []
    # tick_max = float(np.max([v for (_, v) in box_plot_maxes.items()]))
    for i in range(len(results_list)):
        results = results_list[i]
        tick_max = box_plot_maxes[results.modelname]
        # violin_ticks = np.linspace(0.0, tick_max, num=6)
        violin_ticks = np.arange(0.0, tick_max, step=0.5)
        errors = np.sort(results[key])
        modelname = results.modelname
        # shape, loc, scale = scipy.stats.gamma.fit(errors)
        # density_x = np.linspace(errors[0], errors[-1], num=int(round(errors.shape[0]/2)))
        # density_y = scipy.stats.gamma.pdf(density_x, shape, loc=loc, scale=scale)
        # axes_histogram.plot(density_x, density_y, color="black")
        counts, binedges, artist =  axes_histogram.hist(errors, bins=(bins[i] if (type(bins) is list) else bins), label=modelname, density=True, histtype="step")
        binmedians = (binedges[:-1] + binedges[1:])/2.0
        binmedians_list.append(binmedians)
        if vertlines:
            count_maxes.append(float(np.max(counts)))
            vline_vals.append( float(np.percentile(errors, whis[1])) )
            if type(artist)==matplotlib.container.BarContainer:
                vline_colors.append(artist.patches[0].get_facecolor())
            elif type(artist)==matplotlib.patches.Polygon:
                vline_colors.append(artist.get_edgecolor())
            else:
                first_patch = artist[0]
                if type(first_patch)==matplotlib.container.BarContainer:
                    vline_colors.append(first_patch.patches[0].get_facecolor())
                else:
                    vline_colors.append(first_patch.get_edgecolor())

        axes_boxplot = axes_list_boxplot[i]
        axes_violin = axes_list_violinplot[i]
        log_scale=(box_plot_scale=="log")
        scale_kwargs = dict()
        if log_scale:
            print("Using log scale")
            scale_kwargs["base"]=2
        bp_dict = axes_boxplot.boxplot(errors, notch=notch, whis=whis, vert=vert, showfliers=showfliers, showmeans=True, meanline=True)
        # violin_dict = axes_violin.violinplot(errors, vert=vert,  points=300, showextrema=False, showmeans=False, showmedians=False)
        sns.violinplot(data=errors, inner=None, bw_adjust=0.25, split=True, gridsize=300, ax=axes_violin, orient="v" if vert else "h", log_scale=log_scale)
        q0, qf = np.percentile(errors, whis)
        mean = np.mean(errors)
        if vert:
            axes_boxplot.set_ylim(bottom=None if log_scale else 0, top=box_plot_maxes[modelname])
            axes_violin.set_ylim(bottom=None if log_scale else 0, top=box_plot_maxes[modelname])

            axes_boxplot.set_yscale(box_plot_scale, **scale_kwargs)
            # axes_violin.set_yscale(box_plot_scale, **scale_kwargs)

            axes_boxplot.set_title(modelname, fontsize=11)
            axes_violin.set_title(modelname, fontsize=11)
            hline0 = axes_violin.axhline(q0, xmin=0.0, xmax=1.0, label="Inter-quantile Range", color="black", linestyle="--", alpha=0.8)
            hline1 = axes_violin.axhline(qf, xmin=0.0, xmax=1.0, color=hline0.get_color(), linestyle=hline0.get_linestyle(), alpha=hline0.get_alpha())
            meanline = axes_violin.axhline(mean, xmin=0.0, xmax=1.0, label="Mean", color=hline0.get_color(), alpha=hline0.get_alpha())

            axes_boxplot.set_yticks(violin_ticks)
            axes_violin.set_yticks(violin_ticks)
        else:
            axes_violin.yaxis.tick_right()
            axes_violin.yaxis.set_label_position("right")
            axes_boxplot.set_xlim(left=None if log_scale else 0, right=box_plot_maxes[modelname])
            axes_violin.set_xlim(left=None if log_scale else 0, right=box_plot_maxes[modelname])
            

            axes_boxplot.set_xscale(box_plot_scale, **scale_kwargs)
            # axes_violin.set_xscale(box_plot_scale, **scale_kwargs)

            axes_boxplot.set_title(modelname, fontsize=11, x=0.5, y=0.7)
            axes_violin.set_title(modelname, fontsize=11, x=0.5, y=0.7)

            vline0 = axes_violin.axvline(q0, ymin=0.0, ymax=1.0, label="Inter-quantile Range", color="black", linestyle="--", alpha=0.8)
            vline1 = axes_violin.axvline(qf, ymin=0.0, ymax=1.0, color=vline0.get_color(), linestyle=vline0.get_linestyle(), alpha=vline0.get_alpha())
            meanline = axes_violin.axvline(mean, ymin=0.0, ymax=1.0, label="Mean", color=vline0.get_color(), alpha=vline0.get_alpha())

            axes_boxplot.set_xticks(violin_ticks)
            axes_violin.set_xticks(violin_ticks)

    axes_list_violinplot[0].legend( loc="upper right",
                                    bbox_to_anchor = (.975, .75),
                                    fontsize=10.0,
                                    frameon=False, 
                                    fancybox=False, 
                                    numpoints=3)
    

    return (fig_combined_histogram, axes_histogram,),\
            (fig_combined_boxplot, axes_list_boxplot),\
            (fig_combined_violinplot, axes_list_violinplot)
import pandas as pd
def plot_together(results_list : list[PredictionResults], **kwargs):
    vert = kwargs["vert"]
    box_plot_maxes = kwargs["box_plot_maxes"]
    showfliers = kwargs["showfliers"]
    key : str = kwargs["key"]
    whis = kwargs["whis"]
    vertlines = kwargs["vertlines"]
    bins = kwargs["bins"]
    box_plot_scale = kwargs["box_plot_scale"]
    notch = kwargs["notch"]
    scale_ticks = kwargs["scale_ticks"]
    names = np.empty(len(results_list), dtype=object)
    error_arrays = np.empty([len(results_list), results_list[0][key].shape[0]], dtype=results_list[0][key].dtype)
    for (i,res) in enumerate(results_list):
        names[i] = res.modelname
        error_arrays[i, :] = (res[key])[:]
    if not vert:
        names = np.flip(names, axis=0)
        error_arrays = np.flip(error_arrays, axis=0)
    if type(box_plot_maxes) is float:
        box_plot_max = float(box_plot_maxes)
    elif box_plot_maxes is None:
        box_plot_max = np.max([(1.05*float(np.max(res[key])) if showfliers else 1.05*float(np.percentile(res[key], whis[1]))) for res in results_list])
    fig_combined_violinplot, _axes_violin_ = plt.subplots() 
    axes_violin : matplotlib.axes.Axes = _axes_violin_

    fig_combined_boxplot, _axes_boxplot_ = plt.subplots() 
    axes_boxplot : matplotlib.axes.Axes = _axes_boxplot_

    fig_combined_histogram, _axes_histogram_ = plt.subplots() 
    axes_histogram : matplotlib.axes.Axes = _axes_histogram_

    violin_ticks : np.ndarray = np.linspace(0.0, box_plot_max, num=8)
    error_dict = {names[i] : error_arrays[i] for i in range(names.shape[0])}
    dataframe : pd.DataFrame = pd.DataFrame.from_dict(error_dict)
    # sns.violinplot(data=dataframe, inner=None, bw_adjust=0.25, split=False, gridsize=300, cut=0, ax=axes_violin, orient="v" if vert else "h", log_scale=False)
    violin_width = .75
    axes_violin.violinplot([error_arrays[i] for i in range(names.shape[0])], widths=violin_width,
                           bw_method=0.025, vert=vert,  points=300, showextrema=False, showmeans=False, showmedians=False)

    fake_ticks = np.arange(1, len(names) + 1)
    fake_limits = np.asarray([.25, len(names) + 0.75])
    limit_range = fake_limits[1] - fake_limits[0]
    if vert:
        axes_boxplot.set_xlim(left=0, right=box_plot_max)
        axes_violin.set_xlim(fake_limits[0], fake_limits[1])
        axes_violin.set_xticks(fake_ticks, labels=names)
    else:
        axes_boxplot.set_ylim(bottom=0, top=box_plot_max)
        axes_violin.set_ylim(fake_limits[0], fake_limits[1])
        axes_violin.set_yticks(fake_ticks, labels=names)
        axes_boxplot.set_xlim(0.0, box_plot_max)
        axes_violin.set_xlim(0.0, box_plot_max)
    fake_meanplot = axes_violin.plot([],[], color="black", label="Mean")
    fake_quantileplot = axes_violin.plot([],[], color="black", linestyle="--", label="Inter-quantile Range")
    vline_vals = []
    count_maxes = []
    for i in range(names.shape[0]):
        modelname : str = names[i]
        errors = error_arrays[i]
        mean = np.mean(errors)
        q0, qf = np.percentile(errors, whis)
        vline_vals.append(float(qf))
        h0 = float(i+1)
        bottom = h0 - (violin_width/2)*(limit_range/names.shape[0])
        top = h0 + (violin_width/2)*(limit_range/names.shape[0])
        if vert:
            axes_violin.hlines(mean, bottom, top, colors=fake_meanplot[0].get_color(), linestyles=fake_meanplot[0].get_linestyle())
            axes_violin.hlines([q0, qf], bottom, top, colors=fake_quantileplot[0].get_color(), linestyles=fake_quantileplot[0].get_linestyle())
        else:
            axes_violin.vlines(mean, bottom, top, colors=fake_meanplot[0].get_color(), linestyles=fake_meanplot[0].get_linestyle())
            axes_violin.vlines([q0, qf], bottom, top, colors=fake_quantileplot[0].get_color(), linestyles=fake_quantileplot[0].get_linestyle())
    axes_violin.legend()
    return (fig_combined_histogram, axes_histogram, vline_vals, count_maxes),\
            (fig_combined_boxplot, axes_boxplot),\
                (fig_combined_violinplot, axes_violin)
    
def plot_error_histograms(results_list : list[PredictionResults], plotbase : str, 
                          metric="ade", bins=200, notch=True, pad_inches=0.02, box_plot_maxes : None | dict[str,float] = None, vert=False,
                          whis=2.0, combined_subdir="combined", showfliers=True, vertlines : bool = False, scale_ticks : None | np.ndarray = None,
                          formats : Iterable[str] = ["svg",], box_plot_scale = "linear", individual_plots :  bool = False ):
    title = title_dict[metric]
    key = metric
    (fig_combined_histogram, axes_histogram), (fig_combined_boxplot, axes_boxplot), (fig_combined_violinplot, axes_violinplot) =\
      plot_individually(results_list, vert=vert, box_plot_maxes=box_plot_maxes, showfliers = showfliers, key = key, 
                        whis = whis, vertlines = vertlines, bins = bins, box_plot_scale = box_plot_scale, notch = notch, scale_ticks = scale_ticks)
    # (fig_combined_histogram, axes_histogram, vline_vals, count_maxes), (fig_combined_boxplot, axes_boxplot), (fig_combined_violinplot, axes_violinplot) =\
    #   plot_together(results_list, vert=vert, box_plot_maxes=box_plot_maxes, showfliers = showfliers, key = key, 
    #                     whis = whis, vertlines = vertlines, bins = bins, box_plot_scale = box_plot_scale, notch = notch, scale_ticks = scale_ticks)
    savedir = os.path.join(plotbase, combined_subdir)
    if os.path.isdir(savedir):
        shutil.rmtree(savedir)
    os.makedirs(savedir)
    fig_combined_histogram.tight_layout(pad=0.1)
    fig_combined_boxplot.tight_layout(pad=0.1)
    fig_combined_violinplot.tight_layout(pad=0.1)
    if vert:
        boxplot_postfix="vertical"
    else:
        boxplot_postfix="horizontal"
    for file_format in formats:
        _kwargs_ : dict[str] = dict()
        if file_format == "png":
            _kwargs_["backend"] = "agg"
        with plt.rc_context({ "pgf.texsystem": "pdflatex", 'font.family': 'serif', 'text.usetex': file_format == "pgf", 'pgf.rcfonts': False,
                            "savefig.format": file_format, "savefig.bbox" : "tight", "savefig.orientation" : "landscape",
                            "savefig.transparent" : True, "savefig.pad_inches" : pad_inches,
                            "svg.fonttype": 'none', 
                            }) as ctx:
            fig_combined_histogram.savefig(os.path.join(savedir, "histogram"), **_kwargs_)
            fig_combined_boxplot.savefig(os.path.join(savedir, "boxplot_%s" % (boxplot_postfix, )), **_kwargs_)
            fig_combined_violinplot.savefig(os.path.join(savedir, "violinplot_%s" % (boxplot_postfix,)), **_kwargs_)
    custom_scale_helper : CustomScaleHelper = CustomScaleHelper(scale_ticks)
    # custom_scale : matplotlib.scale.FuncScale = matplotlib.scale.FuncScale(axes_histogram, (custom_scale_helper.forward, custom_scale_helper.inverse))
    # axes_violinplot.set_xscale(matplotlib.scale.FuncScale(axes_violinplot, (custom_scale_helper.forward, custom_scale_helper.inverse)))
    # fig_combined_histogram.legend(frameon=False, fancybox=False)
    # axes_histogram.set_xscale(custom_scale) 
    # if vertlines:
    #     print("Plotting vertical lines") #"grey" *float(np.max(count_maxes)) .inverted()
    #     for i in range(len(vline_vals)):
    #         axes_histogram.axvline(vline_vals[i], ymin=0.0, ymax=0.75, color=vline_colors[i], linestyle="--")
    #     overall_max = float(np.max([np.max(res[key]) for res in results_list]))
    #     all_ticks = sorted([0.0,] + vline_vals + [overall_max,])

    # else:
    #     axes_histogram.set_xticks(custom_scale_helper.tickvals)
    #     all_ticks = np.linspace(scale_ticks[0], 1.15*scale_ticks[-1], num=5)
    #     print("Not plotting vertical lines")
    # axes_histogram.set_xticks(all_ticks)
    # axes_histogram.set_xticklabels(["%.2f" % (all_ticks[i],) for i in range(len(all_ticks))], fontsize=11.0)
    
    if not individual_plots:
        return (
            (fig_combined_histogram, axes_histogram),
            (fig_combined_boxplot, axes_boxplot),
            (fig_combined_violinplot, axes_violinplot)
        )
    individual_histogram_figures = []
    individual_boxplot_figures = []
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
        fig.savefig(os.path.join(savedir, "histogram.png"), backend="agg", transparent=True)
        fig.savefig(os.path.join(savedir, "histogram.pdf"), backend="pdf", transparent=True)
        fig.savefig(os.path.join(savedir, "histogram.svg"), backend="svg", transparent=True)
        individual_histogram_figures.append(fig)

        figbox : matplotlib.figure.Figure = plt.figure()
        plt.title(title + ": " + modelname)
        plt.boxplot(errors, notch=notch, whis=whis, showfliers=showfliers)
        figbox.savefig(os.path.join(savedir, "boxplot.png"), backend="agg", transparent=True)
        figbox.savefig(os.path.join(savedir, "boxplot.pdf"), backend="pdf", transparent=True)
        figbox.savefig(os.path.join(savedir, "boxplot.svg"), backend="svg", transparent=True)
        individual_boxplot_figures.append(figbox)
    return (
        (fig_combined_histogram, axes_histogram),
        (fig_combined_boxplot, axes_boxplot),
        (fig_combined_violinplot, axes_violinplot),
        individual_histogram_figures,
        individual_boxplot_figures
    )
        
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
        "color_map" : {res.modelname : None for res in results_list},
        "metric" : "ade",
        "N" : -1,
        "histograms" : True,
        "with_history" : True,
        "ref_alpha" : 1.0,
        "nonref_alpha" : 0.25,
        "pf": None,
        "whis": None,
        "idx_filter" : None,
        "subdir" : None,
        "bins" : "auto",
        # "bins" : 125,
        "notch" : True,
        "showfliers" : False,
        "other_models" : [],
        "vertlines" : False,
        "box_plot_maxes" : None, 
        "scale_ticks" : None,
        "box_plot_scale" : "linear",
        "individual_plots" : False,
        "formats" : ["svg",]
    }
    argdict.update(kwargs)
    color_map : dict[str] = argdict["color_map"]
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
        print("Not filtering")
    try:
        results_trimmed_list : list[PredictionResults] = [results_list[i].subsample(idxgood) for i in range(len(results_list))]
    except:
        results_trimmed_list : list[PredictionResults] = results_list

    dset_trimmed : torchdata.Subset = torchdata.Subset(fulldset, np.where(idxgood)[0])
    # reference_max = float(np.percentile(results_trimmed_list[0][metric], 99))
    if argdict["histograms"]:
        overall_min = 0.975*float(np.min([np.min(res[metric]) for res in results_trimmed_list]))
        overall_max = 1.025*float(np.max([np.max(res[metric]) for res in results_trimmed_list]))
        if type(argdict["scale_ticks"]) is str:
            if argdict["scale_ticks"]=="linear":
                scale_ticks : np.ndarray = np.linspace(overall_min, overall_max, num=8)
            else:
                raise ValueError("\'linear\' is the only supported string value for scale_ticks")
        elif type(argdict["scale_ticks"]) is np.ndarray:
            scale_ticks : np.ndarray = argdict["scale_ticks"]
        else:
            reference_max = 1.025*float(results_trimmed_list[0][metric].max())
            scale_ticks : list[float] = np.linspace(overall_min, reference_max, num=4).tolist()
            scale_ticks.extend(np.linspace(reference_max, overall_max, num=2)[1:].tolist())
            scale_ticks = np.asarray(sorted(list(set(scale_ticks))))
        histogramdir = os.path.join(subdir, "histograms")
        histogramrtn = plot_error_histograms(results_trimmed_list, histogramdir, 
                              whis = whis, metric=metric, showfliers=argdict["showfliers"], box_plot_scale=argdict["box_plot_scale"],
                              box_plot_maxes = argdict["box_plot_maxes"], vertlines=argdict["vertlines"], individual_plots=argdict["individual_plots"],
                              scale_ticks = scale_ticks, bins=argdict["bins"], notch=argdict["notch"], formats=argdict["formats"])
    else:
        histogramrtn = None
    original_summary_dir = os.path.join(subdir, "untrimmed_summaries")
    os.makedirs(original_summary_dir, exist_ok=True)
    trimmed_summary_dir = os.path.join(subdir, "trimmed_summaries")
    os.makedirs(trimmed_summary_dir, exist_ok=True)
    for i in range(len(results_trimmed_list)):
        results = results_list[i]
        with open(os.path.join(original_summary_dir, "%s.yaml" % (results.modelname)), "w") as f:
            yaml.safe_dump(results.error_summary(pf=whis[1]), f, indent=2)
        results_trimmed = results_trimmed_list[i]
        with open(os.path.join(trimmed_summary_dir, "%s.yaml" % (results_trimmed.modelname)), "w") as f:
            yaml.safe_dump(results_trimmed.error_summary(pf=whis[1]), f, indent=2)
    if argdict["N"]<=0:
        return histogramrtn
    plotdir = os.path.join(subdir, "plots")
    plot_outliers(results_trimmed_list, plotdir, dset_trimmed, 
                  N=argdict["N"], worst=True, 
                  with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                  nonref_alpha=argdict["nonref_alpha"])
    plot_outliers(results_trimmed_list, plotdir, dset_trimmed, 
                  N=argdict["N"], worst=False, 
                  with_history=argdict["with_history"], ref_alpha=argdict["ref_alpha"],
                  nonref_alpha=argdict["nonref_alpha"])
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

def plot_outlier_counts(results_list : list[PredictionResults], metric : str, maxval : float, 
                        bar_kw : dict[str] = dict(),  bar_label_kw : dict[str] = dict(), label_axes=False):
    t : tuple[ matplotlib.figure.Figure,  matplotlib.axes.Axes] = plt.subplots()
    fig = t[0]
    ax = t[1]

    outlier_indices = np.stack(
        [results_list[i][metric]>maxval for i in range(len(results_list))]
    , axis=0)
    inlier_indices = ~outlier_indices
    outlier_counts = np.sum(outlier_indices, axis=1)
    xlabels = [results_list[i].modelname for i in range(len(results_list))]
    bar_plot = ax.bar(xlabels, outlier_counts, **bar_kw)
    ax.bar_label(bar_plot, **bar_label_kw)
    if label_axes:
        ax.set_xlabel("Model Name")
        ax.set_ylabel("Outlier Count")
    fig.tight_layout(pad=0.05)
    return fig, ax 

def deframe_axes(ax : matplotlib.axes.Axes, keep_ticks=False, keys=["top", "right", "bottom", "left"]):
    for k in keys:
        ax.spines[k].set_visible(False)
    if not keep_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

def export_legend(axin : matplotlib.axes.Axes, sort_keys : bool = False, **kwargs):
    fig_legend, _ax_legend_ = plt.subplots()
    ax_legend : matplotlib.axes.Axes = _ax_legend_
    ax_legend.set_axis_off()
    handles, keys = axin.get_legend_handles_labels()
    if sort_keys:
        keys_arr = np.asarray(keys, dtype=object)
        idx = np.argsort(keys_arr)
    else:
        idx = np.linspace(0, len(keys) - 1, num = len(keys), dtype=np.int64)
    handles_sorted, keys_sorted = [handles[i] for i in idx], [keys[i] for i in idx]
    handler_map : dict = dict()
    for (i,h) in enumerate(handles_sorted):
        try:
            print(h, h.get_cmap())
            handler_map[h] = HandlerColorLineCollection(numpoints=4)
        except:
            pass
    legend = ax_legend.legend(
        handles_sorted, keys_sorted,
        loc="lower left", 
        bbox_to_anchor=(0, 0),
        frameon=False,
        fancybox=False,
        handler_map = handler_map,
        **{k : v for (k,v) in kwargs.items() if k not in ["loc", "bbox_to_anchor", "frameon", "fancybox"]}
    )
    legend.set_frame_on(False)
    fig_legend.tight_layout(pad = 0.0) #
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    return fig_legend, ax_legend, bbox

import matplotlib.axes
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
import matplotlib.transforms
import deepracing_models.math_utils as mu
def scatter_composite_xy(curve : torch.Tensor, ax : Axes, quiver_kwargs : dict = dict(), **kwargs):
    _colors : torch.Tensor | None = kwargs.get("colors", None)
    if (_colors is None):
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    else:
        colors = _colors
    if curve.shape[2]!=2:
        raise ValueError("Can only plot 2-dimensional curves")
    tswitch : torch.Tensor | None = kwargs.get("tswitch", None)
    tplot : torch.Tensor | None = kwargs.get("tplot", None)
    if (tswitch is not None) and (tplot is not None):
        tstart = tswitch[:-1]
        tend = tswitch[1:]
        dt = tend - tstart
        points_plot, _ = mu.compositeBezierEval(tstart, dt, curve, tplot)
    else:
        points_plot = None
    numsegments = curve.shape[0]
    marker = kwargs.get("marker", "o")
    scatter_kwargs = {k : v for (k,v) in kwargs.items() if k not in ["tswitch", "tannotate", "tarrows", "tplot", "color", "colors", "marker"]}
    plot_kwargs = {k : v for (k,v) in kwargs.items() if k not in ["tswitch", "tannotate", "tarrows", "tplot", "color", "colors", "marker", "label", "s"]}
    ax.scatter(curve[0,:-1,0], curve[0,:-1,1], color=colors[0], marker=marker, **scatter_kwargs)
    if not (points_plot == None):
        idx_segment = tplot<tend[0]
        toplot = torch.cat([curve[0,0].unsqueeze(0), points_plot[idx_segment], curve[0,-1].unsqueeze(0)], dim=0)
        ax.plot(toplot[:,0], toplot[:,1], color=colors[0], label=r"$\mathbf{B}_0$", **plot_kwargs) 
    for i in range(1, numsegments):
        previous_color = colors[(i-1)%len(colors)]
        current_color = colors[i%len(colors)]
        ax.scatter(curve[i,0,0].item(), curve[i,0,1].item(), marker=MarkerStyle(marker, fillstyle='left'), color=previous_color, **scatter_kwargs) 
        ax.scatter(curve[i,0,0].item(), curve[i,0,1].item(), marker=MarkerStyle(marker, fillstyle='right'), color=current_color, **scatter_kwargs) 
        ax.scatter(curve[i,1:,0], curve[i,1:,1], color=current_color, marker=marker, **scatter_kwargs)
        if not (points_plot == None):
            idx_segment = (tplot>=tend[i-1])*(tplot<tend[i])
            toplot = torch.cat([curve[i,0].unsqueeze(0), points_plot[idx_segment], curve[i,-1].unsqueeze(0)], dim=0)
            ax.plot(toplot[:,0], toplot[:,1], color=colors[i], label=r"$\mathbf{B}_" + str(i) + r"$", **plot_kwargs) 
        
    tarrows : torch.Tensor | None = kwargs.get("tarrows", None)
    if tarrows is not None:
        arrow_locs = mu.compositeBezierEval(tstart, dt, curve, tarrows)[0].cpu()
        curve_deriv = (curve.shape[1]-1)*(curve[:,1:] - curve[:,:-1])/dt[...,None,None]
        arrow_vals = mu.compositeBezierEval(tstart, dt, curve_deriv, tarrows)[0].cpu()
        arrow_vals/=torch.norm(arrow_vals, p=2, dim=1, keepdim=True)
        data_to_axes = ax.transLimits
        axes_to_data = data_to_axes.inverted()
        arrowprops = dict(arrowstyle="simple", color="black")
        origin_ax = data_to_axes.transform([0.0, 0.0])
        unitvec_ax = origin_ax + np.asarray([0.0675, 0.0])
        for i in range(arrow_locs.shape[0]):
            px = arrow_locs[i,0].item()
            py = arrow_locs[i,1].item()
            affinemat = np.eye(3)
            affinemat[0,2] = px
            affinemat[1,2] = py
            xaxis = arrow_vals[i].cpu().numpy()
            yaxis = xaxis[[1,0]].copy()
            yaxis[0]*=-1.0
            affinemat[0:2,0] = xaxis
            affinemat[0:2,1] = yaxis
            affine_transform =  matplotlib.transforms.Affine2D(affinemat.copy())
            new_tf = axes_to_data +  affine_transform + ax.transData
            ax.annotate("", unitvec_ax, xycoords=new_tf, xytext=arrow_locs[i], textcoords="data", arrowprops=arrowprops)
        # ax.quiver(arrow_locs[:,0], arrow_locs[:,1], arrow_vals[:,0], arrow_vals[:,1], angles='xy', **quiver_kwargs)
    xmin, xmax = ax.get_xlim()
    dx = xmax - xmin
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin

    return colors, points_plot

def scatter_composite_axes(curve : torch.Tensor, tswitch : torch.Tensor, axes : list[Axes], marker="o", colors : list | None = None, **kwargs):
    if (colors is None):
        _colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    else:
        _colors = colors
    if len(axes)!=curve.shape[2]:
        raise ValueError("Must pass list of axes of equal length to number of dimensions in curve")
    kbezier = curve.shape[1] - 1
    numsegments = curve.shape[0]
    for d in range(curve.shape[2]):
        coefs = curve[:,:,d]
        ax = axes[d]
        t_scatter = torch.linspace(tswitch[0], tswitch[1], steps=kbezier+1, dtype=torch.float64).cpu()#[1:s]
        ax.scatter(t_scatter, coefs[0], color=_colors[0], marker=marker, **kwargs)
        for i in range(1, numsegments):
            t_scatter = torch.linspace(tswitch[i], tswitch[i+1], steps=kbezier+1)[1:].cpu()
            points_scatter = coefs[i,1:].cpu()
            previous_color = _colors[(i-1)%len(_colors)]
            current_color = _colors[i%len(_colors)]
            ax.scatter(t_scatter, points_scatter, color=current_color, marker=marker, **kwargs)
            ax.scatter(tswitch[i].item(), coefs[i,0].item(), marker=MarkerStyle(marker, fillstyle='left'), color=previous_color, **kwargs) 
            ax.scatter(tswitch[i].item(), coefs[i,0].item(), marker=MarkerStyle(marker, fillstyle='right'), color=current_color, **kwargs) 
    return _colors
from scipy.spatial.transform import Rotation
from matplotlib.collections import LineCollection, Collection
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Colormap
import matplotlib.cm
class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans, linestyle=artist.get_linestyle())
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]
def add_colored_line(points : np.ndarray, cvals : np.ndarray, ax : matplotlib.axes.Axes, cmap : str | Colormap, 
    linestyle="solid", alpha=1.0) -> tuple[LineCollection, Collection]:
    points_exp = points.reshape(-1, 1, points.shape[-1])
    segments = np.concatenate([points_exp[:-1], points_exp[1:]], axis=1)
    norm = plt.Normalize(cvals.min(), cvals.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm,linestyle=linestyle, alpha=alpha)
    lc.set_array(cvals)
    line = ax.add_collection(lc)
    return lc, line
def plot_example(ax : matplotlib.axes.Axes, sample : dict[str,np.ndarray], 
                 predictions : np.ndarray | None = None, 
                 cmap: str | Colormap | None = None,
                 colorbar_label: str | None = None,
                 rotation : Rotation = Rotation.identity()):
    mask = np.ones_like(sample["future_left_bd"])
    mask[:,-1] = 0.0
    future_left_bd = rotation.apply(sample["future_left_bd"]*mask)[:,[0,1]]
    future_right_bd = rotation.apply(sample["future_right_bd"]*mask)[:,[0,1]]
    ground_truth = rotation.apply(sample["fut"]*mask)[:,[0,1]]
    ground_truth_vel = rotation.apply(sample["fut_vel"]*mask)[:,[0,1]]
    ground_truth_speeds = np.linalg.norm(ground_truth_vel, ord=2.0, axis=1)
    all_points = [future_left_bd, future_right_bd, ground_truth]
    lbartists, = ax.plot(future_left_bd[:,0], future_left_bd[:,1], alpha=1.0, linestyle="solid", color="black", label="Track Boundaries")
    rbartists, = ax.plot(future_right_bd[:,0], future_right_bd[:,1], alpha=lbartists.get_alpha(), linestyle=lbartists.get_linestyle(), color=lbartists.get_color())
    if cmap is not None:
        gt_lc, gt_line = add_colored_line(ground_truth, ground_truth_speeds, ax, "RdYlGn")
        gt_line.set_label("Ground Truth")
        norm = plt.Normalize(ground_truth_speeds.min(), ground_truth_speeds.max(), clip=True)
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = ax.get_figure().colorbar(scalar_mappable, ax=ax, location='left', pad=0.01, label=colorbar_label, shrink=0.8)
        ticks = np.linspace(ground_truth_speeds.min(), ground_truth_speeds.max(), num=2)
        cb.set_ticks(ticks, labels=["%3.2f" %(float(tick),) for tick in ticks])
        rtn = gt_lc, gt_line
    else:
        gtartists, = ax.plot(ground_truth[:,0], ground_truth[:,1], color="grey", label="Ground Truth")
        rtn = None, None
        # gtartists = ax.scatter(ground_truth[:,0], ground_truth[:,1], color="grey", label="Ground Truth", s=2.0**3.0, alpha=0.6)

    if predictions is not None:
        predictions_2d = predictions[:,[0,1]].copy()
        predictions_3d = np.concatenate([predictions_2d, np.zeros_like(predictions_2d[:,[0,]])], axis=1)
        predictions = rotation.apply(predictions_3d)[:,[0,1]]
        predictionartist, = ax.plot(predictions[:,0], predictions[:,1], label="Predictions", alpha=1.0)
        all_points.append(predictions)

    all_points = np.concatenate(all_points, axis=0)
    min_x, max_x = np.min(all_points[:,0]) - 0.0, np.max(all_points[:,0]) + 0.0
    min_y, max_y = np.min(all_points[:,1]) - 0.0, np.max(all_points[:,1]) + 0.0
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    # aspect_ratio, adjustable, anchor = (max_x - min_x)/(max_y - min_y), "box", "NW"
    # aspect_ratio, adjustable, anchor = "equal", "box", "NW"
    aspect_ratio, adjustable, anchor = "auto", None, None

    ax.set_aspect(aspect_ratio, adjustable = adjustable, anchor = anchor)