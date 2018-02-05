import os
import numpy as np


class Numpy_dataset(object):

    def __init__(self, data_dir, file_name='train.npz'):

        file_path = os.path.join(data_dir,file_name)
        with np.load(file_path) as data:
            self.X = data['X']
            self.y = data['y'].astype(np.uint8)
            self.samples_names = data["samples_names"]
            self.groups2 = data["fold_number"]
            self.groups = data["fold_session"]
        if len(self.X.shape) == 3:
            self.X = self.X.reshape(-1,1,self.X.shape[1],self.X.shape[2])

        self.y = self.y - self.y.min()
        self.nclasses = self.y.max()+1

