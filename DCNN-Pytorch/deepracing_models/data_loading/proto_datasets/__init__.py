from .PoseSequenceDataset import PoseSequenceDataset
from .ControlOutputDataset import ControlOutputDataset
from .ControlOutputSequenceDataset import ControlOutputSequenceDataset
import os
import deepracing.backend
import numpy as np

def load_sequence_label_datasets(dataset_config : dict, model_config : dict):
    dsets=[]
    use_optflow = dataset_config.get("use_optflow",False)
    max_spare_txns = dataset_config.get("max_spare_txns",50)
    dsetfolders = []
    image_size = dataset_config["image_size"]
    context_length = model_config["context_length"]
    for dataset in dataset_config["datasets"]:
        print("Parsing database config: %s" %(str(dataset)))
        root_folder = dataset["root_folder"]
        dsetfolders.append(root_folder)
        label_folder = os.path.join(root_folder,"pose_sequence_labels")
        image_folder = os.path.join(root_folder,"images")
        key_file = os.path.join(root_folder,"goodkeys.txt")
        apply_color_jitter = dataset.get("apply_color_jitter",False)
        erasing_probability = dataset.get("erasing_probability",0.0)
        label_wrapper = deepracing.backend.PoseSequenceLabelLMDBWrapper()
        label_wrapper.readDatabase(os.path.join(label_folder,"lmdb"), max_spare_txns=max_spare_txns )
        image_mapsize = float(np.prod(image_size)*3+12)*float(len(label_wrapper.getKeys()))*1.1

        image_wrapper = deepracing.backend.ImageLMDBWrapper(direct_caching=False)
        image_wrapper.readDatabase(os.path.join(image_folder,"image_lmdb"), max_spare_txns=max_spare_txns, mapsize=image_mapsize )


        curent_dset = PoseSequenceDataset(image_wrapper, label_wrapper, key_file, context_length,\
                     image_size = image_size, return_optflow=use_optflow, apply_color_jitter=apply_color_jitter, erasing_probability=erasing_probability)
        dsets.append(curent_dset)
        print("\n")
    return dsets