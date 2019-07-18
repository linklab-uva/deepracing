import torch
import torch.utils.data as data_utils
import data_loading.proto_datasets
from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser(description="Dataset Playground")
parser.add_argument("dataset_dir", type=str,  help="Proto directory to load")
parser.add_argument("--processes", type=int, default=0,  help="Number of extra processes to use")
args = parser.parse_args()
print("Hello World!")
#dset = h5utils.DeepRacingH5Dataset(args.dataset_file, map_entire_file = args.map_entire_file)
dset = data_loading.proto_datasets.ProtoDirDataset(args.dataset_dir, 10, 5)



#image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time = dset[0]
batch_size=4
dataloader = data_utils.DataLoader(dset, batch_size=batch_size,
                        shuffle=True, num_workers=args.processes)
t = tqdm(enumerate(dataloader))
for i_batch, (image_torch, position_torch, rotation_torch, linear_velocity_torch, angular_velocity_torch, session_time) in t:
   pass