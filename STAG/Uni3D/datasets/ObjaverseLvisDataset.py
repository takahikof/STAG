import os
import torch
import numpy as np
import h5py
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ObjaverseLvisDataset(data.Dataset):

    label2id_dict = None # class variable

    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.N_POINTS
        self.permutation = np.arange(self.npoints)
        self.use_color=config.USE_COLOR

        if( self.subset == "train" ):
            h5_filepath = self.data_root + "/objaverse_lvis_2048pts_withcolor_train.h5"
        elif( self.subset == "test" ):
            h5_filepath = self.data_root + "/objaverse_lvis_2048pts_withcolor_test.h5"
        else:
            raise NotImplementedError()

        h5 = h5py.File( h5_filepath, 'r')
        self.points = np.array(h5['data']).astype(np.float32)
        self.labels = np.array(h5['label']).astype(int)
        h5.close()

        print_log(f'[DATASET] {self.points.shape[0]} instances were loaded', logger = 'ObjaverseLVIS')

        # Convert label name (non-sequential intergers) to label ID (sequential intergers)
        if( self.label2id_dict is None ):
            if( self.subset != "train" ):
                print( "error: training set must be loaded first." )
                quit()
            self.__class__.label2id_dict = {}
            label_names = np.unique( self.labels )
            for i in range( label_names.size ):
                self.__class__.label2id_dict[ label_names[ i ] ] = i
            # print( self.label2id_dict )


    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc


    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        data  = self.points[idx]
        class_id = self.labels[idx]

        if not self.use_color:
            data=data[:,0:3]

        data = self.random_sample(data, self.sample_points_num)
        data[:,0:3] = self.pc_norm(data[:,0:3])

        data = torch.from_numpy(data).float()

        class_id = self.label2id_dict[ class_id ]

        return 'ObjaverseLVIS', 'sample', ( data, class_id )

    def __len__(self):
        return self.points.shape[0]
