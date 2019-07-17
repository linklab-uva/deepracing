import torch
import torch.utils.data as data_utils
import h5utils
import argparse
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser(description="Dataset Playground")
parser.add_argument("dataset_file", type=str,  help="H5 file to use")
parser.add_argument("--processes", type=int, default=0,  help="Number of extra processes to use")
parser.add_argument("--map_entire_file", action="store_true")
args = parser.parse_args()

dset = h5utils.DeepRacingH5Dataset(args.dataset_file, map_entire_file = args.map_entire_file)

image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time = dset[0]
dataloader = data_utils.DataLoader(dset, batch_size=4,
                        shuffle=True, num_workers=args.processes)
t = tqdm(enumerate(dataloader))
for i_batch, (image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time) in t:
    t.set_postfix({"index":i_batch , "image_shape": image_torch.shape,"rotation_shape" : rotation_torch.shape})