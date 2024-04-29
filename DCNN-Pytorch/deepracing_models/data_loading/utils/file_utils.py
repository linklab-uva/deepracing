import numpy as np
import yaml
import glob
import os
import deepracing_models.data_loading.file_datasets as FD
from deepracing_models.data_loading import SubsetFlag
import torch


def load_datasets_from_files(search_dir : str, 
                             keys = FD.TrajectoryPredictionDataset.KEYS_WE_CARE_ABOUT, 
                             kbezier : int | None = None,
                             segments : int = 1,
                             constrain_tangents : bool = True, 
                             bcurve_cache = False, 
                             flag = SubsetFlag.TRAIN,
                             dtype=np.float64,
                             device=torch.device("cpu")):
    def sortkey(filepath : str):
        with open(filepath, "r") as f:
            metadata : dict = yaml.load(f, Loader=yaml.SafeLoader)
        real_data = metadata.get("real_data", False)
        if real_data:
            source_bag : str = metadata["source_bag"]
            source_topic : str = metadata["source_topic"]
            return os.path.basename(source_bag).replace("/","_"), source_topic[1:].replace("/","_")
        else:
            subfolder = os.path.dirname(filepath)
            subfolder_base = os.path.basename(subfolder)
            bagfolder = os.path.dirname(subfolder)
            bagfolder_base = os.path.basename(bagfolder)
            car_index = subfolder_base.split("_")[1]
            return bagfolder_base, int(car_index)
    
    dsetfiles = []
    for t in os.walk(search_dir):
        dirpath : str = t[0] 
        dirnames : list[str] = t[1]
        filenames : list[str] = t[2]
        if "DEEPRACING_IGNORE" in filenames:
            dirnames.clear()
            continue
        try:
            dirnames.remove("plots")
        except ValueError as e:
            pass
        try:
            dirnames.remove("fit_data")
        except ValueError as e:
            pass
        if "metadata.yaml" in filenames:
            fullpath = os.path.join(dirpath, "metadata.yaml")
            # dsetfiles.append(fullpath)
            # dirnames.clear()
            with open(fullpath, "r") as f:
                configdict = yaml.safe_load(f)
            if configdict.get("DEEPRACING_DATASET", False):
                dsetfiles.append(fullpath)
                dirnames.clear()
    # dsetfiles = glob.glob(os.path.join(search_dir, "**", "metadata.yaml"), recursive=True)
    dsetfiles.sort(key=sortkey)
    dsets : list[FD.TrajectoryPredictionDataset] = []
    dsetconfigs = []
    numsamples_prediction = None
    for metadatafile in dsetfiles:
        with open(metadatafile, "r") as f:
            dsetconfig = yaml.load(f, Loader=yaml.SafeLoader)
        if numsamples_prediction is None:
            numsamples_prediction = dsetconfig["numsamples_prediction"]
        elif numsamples_prediction!=dsetconfig["numsamples_prediction"]:
            raise ValueError("All datasets must have the same number of prediction points. " + \
                            "Dataset at %s has prediction length %d, but previous dataset " + \
                            "has prediction length %d" % (metadatafile, dsetconfig["numsamples_prediction"], numsamples_prediction))
        dsetconfigs.append(dsetconfig)
        dsets.append(FD.TrajectoryPredictionDataset.from_file(metadatafile, flag, dtype=dtype, keys=keys))
        if kbezier is not None:
            dsets[-1].fit_bezier_curves(kbezier, device=device, cache=bcurve_cache, segments=segments, constrain_tangents=constrain_tangents)
    return dsets

def load_datasets_from_shared_memory(
        shared_memory_locations : list[ tuple[ dict[str, tuple[str, list]], dict ]  ],
        dtype : np.dtype
    ):
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for shm_dict, metadata_dict in shared_memory_locations:
        dsets.append(FD.TrajectoryPredictionDataset.from_shared_memory(shm_dict, metadata_dict, SubsetFlag.TRAIN, dtype=dtype))
    return dsets