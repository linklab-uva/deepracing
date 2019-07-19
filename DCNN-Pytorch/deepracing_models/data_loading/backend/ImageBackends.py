from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
import os
from skimage.transform import resize
import imutils
class LMDBWrapper():
    def __init__(self):
        self.txn = None
        self.env = None
        self.im_size = None
    def readImages(self, image_files, keys, db_path, im_size=None):
        assert(len(image_files) > 0)
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        self.env = lmdb.open(db_path, map_size=1e10)
        size_specified = im_size is not None
        if size_specified:
            self.im_size = im_size.astype(np.int16)
        else:
            self.im_size = np.array(skimage.io.imread(image_files[0]).shape).astype(np.int16)
        with self.env.begin(write=True) as write_txn:
            print("Loading image data")
            write_txn.put("imsize".encode("ascii"), self.im_size.tobytes())
            for i, key in tqdm(enumerate(keys)):
                if size_specified:
                    im = imutils.resizeImage(skimage.util.img_as_ubyte(skimage.io.imread(image_files[i])), self.im_size[0:2])
                else:
                    im = skimage.util.img_as_ubyte(skimage.io.imread(image_files[i]))
                write_txn.put(key.encode("ascii"), im.flatten().tobytes())
        self.txn = self.env.begin(write=False)
    def readDatabase(self, db_path : str):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=1e6)
        self.txn = self.env.begin(write=False)
        self.im_size = np.fromstring(self.txn.get("imsize".encode("ascii")), dtype=np.uint16)
    def getImage(self, key):
        return np.reshape(np.fromstring(self.txn.get(key.encode("ascii")), dtype=np.uint8), self.im_size)
