from torch import utils
import numpy as np


class Datasets(utils.data.Dataset):
    
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

        self.datanum = len(self.X)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_X = self.X[idx]
        out_y = self.y[idx]

        # (64, 64, 5) -> (5, 64, 64)
        out_X = out_X.transpose((2, 0, 1))

        return out_X, out_y
