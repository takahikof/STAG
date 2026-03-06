import os
import numpy as np
import warnings
import pickle

from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

@DATASETS.register_module()
class Grocery100(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.num_category = config.NUM_CATEGORY
        split = config.subset
        self.subset = config.subset
        self.use_color=config.USE_COLOR

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))

        print_log('Load processed data from %s...' % self.save_path, logger = '3DGrocery100')
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

        print_log('The size of %s data is %d' % (split, len( self.list_of_points )), logger = '3DGrocery100')


    def __len__(self):
        return len( self.list_of_points )

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_color:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])   # 1024
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return '3DGrocery100', 'sample', (current_points, label)
