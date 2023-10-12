import numpy as np
import yaml
import glob
import os
import deepracing_models.data_loading.file_datasets as FD
from deepracing_models.data_loading import SubsetFlag


def load_datasets_from_files(search_dir : str, keys = FD.TrajectoryPredictionDataset.KEYS_WE_CARE_ABOUT, kbezier : int | None = None, bcurve_cache = False, dtype=np.float64):
    def sortkey(filepath : str):
        subfolder = os.path.dirname(filepath)
        bagfolder = os.path.dirname(subfolder)
        subfolder_base = os.path.basename(subfolder)
        car_index = subfolder_base.split("_")[1]
        bagfolder_base = os.path.basename(bagfolder)
        return bagfolder_base, int(car_index)
    dsetfiles = glob.glob(os.path.join(search_dir, "**", "metadata.yaml"), recursive=True)
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
        dsets.append(FD.TrajectoryPredictionDataset.from_file(metadatafile, SubsetFlag.TRAIN, dtype=dtype, keys=keys))
        if kbezier is not None:
            dsets[-1].fit_bezier_curves(kbezier, cache=bcurve_cache)
    return dsets

def load_datasets_from_shared_memory(
        shared_memory_locations : list[ tuple[ dict[str, tuple[str, list]], dict ]  ],
        dtype : np.dtype
    ):
    dsets : list[FD.TrajectoryPredictionDataset] = []
    for shm_dict, metadata_dict in shared_memory_locations:
        dsets.append(FD.TrajectoryPredictionDataset.from_shared_memory(shm_dict, metadata_dict, SubsetFlag.TRAIN, dtype=dtype))
    return dsets