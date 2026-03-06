import os
import torch
import numpy as np
import h5py
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class MCB_B(data.Dataset):
    def __init__(self, config):

        self.subset = config.subset
        self.root = config.ROOT
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'MCB_B_2048pts_withnormal_train.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'MCB_B_2048pts_withnormal_test.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print_log('The number of data: %d' % self.points.shape[0], logger = 'MCB_B')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        pos = pc[:, 0:3 ]
        ori = pc[:, 3:6 ]
        centroid = np.mean( pos, axis=0 )
        pos = pos - centroid
        m = np.max(np.sqrt(np.sum(pos**2, axis=1)))
        pos = pos / m
        pc = np.hstack( [ pos, ori ] )
        return pc

    def __getitem__(self, index):

        point_set = self.points[ index ]
        label = self.labels[ index ]

        point_set = self.pc_norm( point_set )

        # resample points
        if( self.npoints < point_set.shape[0] ):
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[choice, :]

        # shuffle points
        permutation = np.arange(self.npoints)
        np.random.shuffle(permutation)
        point_set = point_set[permutation]

        if ( not self.use_normals ):
            point_set = point_set[:,0:3]

        return 'MCB_A', 'sample', (point_set, label)

    def __len__(self):
        return self.points.shape[0]
