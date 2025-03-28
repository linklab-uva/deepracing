import pandas as pd
import os
import torch
import yaml
import pickle
from typing import Any, Iterable
from deepracing_models.math_utils import RacelineHelper
import tqdm
def flatten_dict(d : dict, sep : str= "."):
    dflattened = dict()
    for (k, v) in d.items():
        if type(v)==dict:
            for (subk, subv) in v.items():
                dflattened[k+sep+subk]=subv
        elif type(v)==list:
            for i in range(len(v)):
                dflattened[k+"_%d" % (i,)] = v[i]
        else:
            dflattened[k]=v
    return dflattened
class DBFExperiment:
    def __init__(self, maindir : str):
        self.maindir = maindir
        self.rootdir = os.path.abspath(os.path.join(maindir, "..", "..", ".."))
        
        with open(os.path.join(maindir, "metadata.yaml"), "r") as f:
            metadata_raw : dict[str, str | int | float | dict]  = yaml.safe_load(f)
        self.metadata : dict[str, str | int | float] = flatten_dict(metadata_raw)
        
        with open(os.path.join(maindir, "kwargs.yaml"), "r") as f:
            kwargs_raw : dict[str, str | int | float | dict]  = yaml.safe_load(f)
        self.kwargs : dict[str, str | int | float] = flatten_dict(kwargs_raw)
        self.kwargs["raceline_offset"] = self.kwargs.get("raceline_offset", 0.0)
        self.kwargs.pop("savedir", "asdf")
    def __str__(self):
        return "DBF Experiment: "+str(self.maindir) +"\n"+str(self.metadata)
    def datadict(self):
        with open(os.path.join(self.maindir, "data.pt"), "rb") as f:
            d : dict[str,torch.Tensor] = torch.load(f)
        return d
    def filter_params(self):
        with open(os.path.join(self.maindir, "filter_params.pt"), "rb") as f:
            d : dict[str,torch.nn.Parameter] = torch.load(f)
        return d
    def raceline_params(self):
        with open(os.path.join(self.maindir, "raceline_helper.pt"), "rb") as f:
            d : dict[str,torch.nn.Parameter] = torch.load(f)
        return d
    def raceline_helper(self):
        raceline_params = self.raceline_params()
        Nrlpoints = int(raceline_params["__times_in__"].shape[0])
        rlorder = int(raceline_params["__curve_r__.control_points"].shape[-2])-1
        rlambientdim = int(raceline_params["__curve_r__.control_points"].shape[-1])

        curve_control_points = torch.zeros([Nrlpoints-1, rlorder+1, rlambientdim]).type_as(raceline_params["__curve_r__.control_points"])
        times_in = torch.linspace(0.0, 100.0, Nrlpoints).type_as(curve_control_points)
        arclengths_in = times_in.clone()
        r_of_t_coefs = torch.zeros_like(curve_control_points[...,:-1,[0,]])
        speed_of_r_coefs = torch.zeros_like(curve_control_points[...,[0,]])


        raceline_helper : RacelineHelper = RacelineHelper(arclengths_in, times_in, curve_control_points, speed_of_r_coefs, r_of_t_coefs)
        raceline_helper.load_state_dict(raceline_params)
        
        return raceline_helper
class DBFExperimentCollection:
    def __init__(self, exeriments : list[DBFExperiment]) -> None:
        self.experiments = sorted(exeriments, key = DBFExperimentCollection.__sortkey__)
    @staticmethod
    def __sortkey__(x : DBFExperiment) -> tuple[str | int | float, str | int | float]:
        return (x.kwargs["trackname"], x.metadata["t0"])
    def __len__(self) -> int:
        return len(self.experiments)
    def __getitem__(self, idx : int) -> dict[str, Any]:
        return {
            "kwargs" : self.experiments[idx].kwargs,
            "metadata" : self.experiments[idx].metadata,
            "data" : self.experiments[idx].datadict(),
        }
    def extend(self, other : "DBFExperimentCollection") -> None:
        self.experiments.extend(other.experiments)
        self.experiments.sort(key = DBFExperimentCollection.__sortkey__)
    def to_pickle(self, filename : str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self.experiments, f)
    @staticmethod
    def from_pickle(filename : str) -> "DBFExperimentCollection":
        with open(filename, "rb") as f:
            experiments = pickle.load(f)
        return DBFExperimentCollection(experiments)
    def to_dataframe(self, metadatakeys : Iterable[str], kwargkeys : Iterable[str]) -> pd.DataFrame:
        points = []
        for exp in self.experiments:
            point = {k : exp.metadata[k] for k in metadatakeys}
            # for k in kwargkeys:
            #     val = exp.kwargs[k]
            #     if type(val)==list:
            #         for i in range(len(val)):
            #             point[k+"_%d" % (i,)] = val[i]
            #     else:
            #         point[k] = val
            point.update({k : exp.kwargs[k] for k in kwargkeys})
            point["maindir"] = exp.maindir
            point["rootdir"] = exp.rootdir
            points.append(point)
        return pd.DataFrame(points)
def cache_experiment_data(rootdir : str, mandatory_keys : set[str] | None = None, cachefile : str | None = None) -> DBFExperimentCollection:
    if mandatory_keys is None:
        with open(os.path.join(rootdir, "default_args.yaml"), "r") as f:
            default_args = flatten_dict(yaml.safe_load(f))
        default_args.pop("savedir", None)
        mandatory_keys = set(default_args.keys())
    experiment_dirs = []
    rootdir_abs=os.path.abspath(rootdir)
    t = tqdm.tqdm(os.walk(rootdir_abs), ncols=300)
    t.set_description("Searching for experiments in %s" % (rootdir_abs,))
    for asdf in t:
        dirpath : str = asdf[0]
        dirnames : list[str] = asdf[1]
        filenames : list[str] = asdf[2]
        if {"DBF_OVERTAKING_ABLATION", "metadata.yaml", "kwargs.yaml"}.issubset(set(filenames)):
            dirnames.clear()
            experiment_dirs.append(dirpath)
        t.set_postfix({"Experiments Found": len(experiment_dirs)})
    experiments = []
    t = tqdm.tqdm(experiment_dirs)
    for experiment_dir in t:
        exp_to_add = DBFExperiment(experiment_dir)
        if mandatory_keys.issubset(set(exp_to_add.kwargs.keys())):
            experiments.append(exp_to_add)  
        t.set_postfix({"Valid Experiments Found": len(experiments)})
    newcollection = DBFExperimentCollection(experiments)
    newcollection.to_pickle(os.path.join(rootdir, "cache.pkl") if (cachefile is None) else cachefile)
    return newcollection