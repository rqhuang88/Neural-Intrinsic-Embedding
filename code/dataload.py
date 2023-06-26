from ssl import OP_ENABLE_MIDDLEBOX_COMPAT
import numpy as np
import warnings
from torch.utils.data import Dataset
import potpourri3d as pp3d
import torch
from utils import *
warnings.filterwarnings('ignore')
import scipy
import scipy.io
import hdf5storage
import os




class SCAPE_5K_test_DataLoader(Dataset):
    def __init__(self, root, split='test', num_of_points = 3000):
        self.split  = split
        self.root = root
        self.sample_num = num_of_points
        if self.split == 'train':
            pass
        elif  self.split == 'test':
            self.raw_file_names = ["mesh%03d.off" % fi for fi in range(72)][52:]
            self.FPS = np.load("data/SCAPE_5k/test_FPS.npy")

    def __len__(self):
        return len(self.raw_file_names)


    def _get_item(self, index):
        point_set , _  = pp3d.read_mesh(self.root  +'/%s' % self.raw_file_names[index])
        point_set =  torch.from_numpy(point_set).float().cuda()
        point_set = pc_normalize(point_set)
        far_pts_point =  self.FPS[index]
        far_pts_point = torch.from_numpy(far_pts_point[:self.sample_num]).long().cuda()
        point_set_far = point_set[far_pts_point]

        return point_set,far_pts_point,point_set_far
        
        
    def __getitem__(self, index):
        return self._get_item(index)


class FAUST_5K_test_DataLoader(Dataset):
    def __init__(self, root, split='test', num_of_points = 3000):
        self.split  = split
        self.root = root
        self.sample_num = num_of_points
        np.random.seed(0)
        
        if self.split == 'train':
            pass
        elif  self.split == 'test':
            self.raw_file_names = ["tr_reg_%03d.off" % fi for fi in range(100)][80:]
            self.FPS =self.FPS = np.load("data/FAUST_5k/test_FPS.npy")

    def __len__(self):
        return len(self.raw_file_names)


    def _get_item(self, index):
        point_set , _  = pp3d.read_mesh(self.root  +'/%s' % self.raw_file_names[index])
        point_set =  torch.from_numpy(point_set).float().cuda()
        point_set = pc_normalize(point_set)
        far_pts_point =  self.FPS[index]
        far_pts_point = torch.from_numpy(far_pts_point[:self.sample_num]).long().cuda()
        point_set_far = point_set[far_pts_point]

        return point_set,far_pts_point,point_set_far
        
        
    def __getitem__(self, index):
        return self._get_item(index)


