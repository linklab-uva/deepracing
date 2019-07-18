from tqdm import tqdm as tqdm
import numpy as np
import skimage
import lmdb
from skimage.transform import resize
class LMDBWrapper():
    def __init__(self, im_size):
        self.txn = None
        self.env = None
        self.im_size = im_size
    def readImages(self, image_files, keys, db_path):
        assert(len(image_files) == len(keys))
        if os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is already a directory")
        os.makedirs(db_path)
        env = lmdb.open(db_path, map_size=1e9)
        with env.begin(write=True) as write_txn:
            print("Loading image data")
            for i in range(keys):
                im = skimage.util.img_as_ubyte(resize(skimage.io.imread(image_files[i]), self.im_size))
                write_txn.put(keys[i], im.flatten().tobytes())
        self.env = lmdb.open(db_path, map_size=1e6)
    def readDatabase(self, db_path : str):
        if not os.path.isdir(db_path):
            raise IOError("Path " + db_path + " is not a directory")
        self.env = lmdb.open(db_path, map_size=1e6)
    def getImage(self, key):
        with env.begin(write=False) as read_txn:
            return np.reshape(np.fromstring(read_txn.get(key), dtype=np.uint8), self.im_size)
