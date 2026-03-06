import os
import torch
import numpy as np
import glob
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ESTATE(data.Dataset):

    ctg2id = {}

    def __init__(self, config):

        self.subset = config.subset
        self.root = config.ROOT
        self.npoints = config.N_POINTS
        self.use_color = config.USE_COLOR

        # Create dictionary that maps category name to category ID
        if( self.ctg2id == {} ):
            ctglist_file = self.root + "/XYZ/estate_shape_names.txt"
            with open( ctglist_file ) as f:
                ctglist = f.read().splitlines()
            for i in range( len( ctglist ) ):
                self.__class__.ctg2id[ ctglist[ i ] ] = i

        # Load point cloud files
        self.points = []
        self.labels = []
        print_log( "Loading " + self.subset + " set...", logger = 'ESTATE' )
        pcdlist_file = self.root + "/XYZ/estate_" + self.subset + ".txt"
        with open( pcdlist_file ) as f:
            pcdlist = f.read().splitlines()

        for pcd in pcdlist:
            ctg_name = pcd.split("_")[0]
            pcd_file = self.root + "/XYZ/" + ctg_name + "/" + pcd + ".txt"
            if( os.path.exists( pcd_file ) ):
                pcd = np.loadtxt( pcd_file, delimiter=",", dtype=np.float32 )
                self.points.append( pcd )
                self.labels.append( self.ctg2id[ ctg_name ] )
            else:
                print( pcd_file + " doesn't exist." )

        print_log('The number of data: %d' % len( self.points ), logger = 'ESTATE')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, index):

        point_set = self.points[ index ]
        label = self.labels[ index ]

        point_set[:, 0:3] = self.pc_norm( point_set[:, 0:3] )

        # resample points
        if( self.npoints < point_set.shape[0] ):
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            point_set = point_set[choice, :]
        else:
            N_points_lacking = self.npoints - point_set.shape[0]
            idx = np.random.randint( low=0, high=point_set.shape[0], size=N_points_lacking )
            point_set = np.vstack( [ point_set, point_set[ idx ] ] )

        # shuffle points
        permutation = np.arange(self.npoints)
        np.random.shuffle(permutation)
        point_set = point_set[permutation]

        if ( not self.use_color ):
            point_set = point_set[:,0:3]

        # print( point_set.shape )
        # point_set = point_set.astype( np.float32 )

        return 'ESTATE', 'sample', (point_set, label)

    def __len__(self):
        return len( self.points )
