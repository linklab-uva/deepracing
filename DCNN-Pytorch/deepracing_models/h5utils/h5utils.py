import h5py
import numpy as np
import os
def imageFolderToDataset(image_folder : str, output_file : str = os.path.join(image_folder,"dataset.hdf5", use_json=True)):
   if use_json:
      image_filepaths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) \
                        if os.path.isfile(os.path.join(image_folder, f)) and str.lower(os.path.splitext(f)[1])==".json"]  
      for jsonstring in image_jsonstrings:
         image_data = TimestampedImage_pb2.TimestampedImage()
         try:
            google.protobuf.json_format.Parse(jsonstring, image_data)
            image_tags.append(image_data)
         except:
            continue
   else:
      image_filepaths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) \
                        if os.path.isfile(os.path.join(image_folder, f)) and str.lower(os.path.splitext(f)[1])==".pb"]
   h5_file = h5py.File(output_file, "w")
   image_dset = h5_file.create_dataset("images")
   label_dset = h5_file.create_dataset("labels")
   h5_file.close()