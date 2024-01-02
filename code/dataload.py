from pickle import NONE
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

def DataLoader_set(params,split):
    if params.DataLoader == 'SCAPE_5K_train_DataLoader':
        return SCAPE_5K_train_DataLoader(params.root,params.vts,params.geod,split,params.num_of_points)


class SCAPE_5K_train_DataLoader(Dataset):
    def __init__(self, root, vts_dir, geod_dir,split='train',num_of_points = 2000):
        self.split  = split
        self.root = root
        self.vts_dir = vts_dir
        self.batch_num = 4995 # in order to keep the same point number for each batch, do net need in test
        self.num_of_points = num_of_points # for loss calculation
        self.sample_num = 3000 # for FPS sampling in DGCNN
        self.geod_dir = geod_dir #YOUR Geodesic distance PATH#
        if self.split == 'train':
            self.raw_file_names = ["mesh%03d.off" % fi for fi in range(72)][:51]
            self.geod_file_names = ["mesh%03d.npy" % fi for fi in range(72)][:51]
            self.EPOCH_counter = torch.zeros(51)
            self.num_of_points = num_of_points
            
        else:
            self.raw_file_names = ["mesh%03d.off" % fi for fi in range(72)][52:] 
            self.geod_file_names = ["mesh%03d.npy" % fi for fi in range(72)][52:] 
            self.EPOCH_counter = torch.zeros(20)
            self.num_of_points = self.batch_num 

    def __len__(self):
        return len(self.raw_file_names)

    def _get_item(self, index):
        point_set , _  = pp3d.read_mesh(self.root  +'/%s' % self.raw_file_names[index])
        far_pts_point = farthest_point_sample(point_set, point_set.shape[0])
        point_set = torch.from_numpy(point_set).float().cuda()
        point_set = pc_normalize(point_set)
      
        geod_full = np.load(os.path.join(self.geod_dir  +'/%s' % self.geod_file_names[index]))
        geod_full  = torch.from_numpy(geod_full).float().cuda()

        
        far_pts_point = torch.from_numpy(far_pts_point).long().cuda()[:self.batch_num]
        point_set = point_set[far_pts_point]
        point_set_grid = point_set[:self.sample_num]
        geod_full = geod_full[far_pts_point]
        geod_full = geod_full.transpose(1,0)[far_pts_point]
        far_pts_point_3k = torch.arange(0,self.sample_num).long()
        far_pts =  torch.arange(0,self.num_of_points).long().cuda()
        pts_1 = torch.tensor(point_set[far_pts[0::2]]).cuda()
        pts_2 = torch.tensor(point_set[far_pts[1::2]]).cuda()
        knn_idx_12,_ = knnsearch(pts_2,pts_1,1)
        knn_idx_21,_ = knnsearch(pts_1,pts_2,1)
        geod_sapmle = geod_full[far_pts]
    
        self.EPOCH_counter[index] += 1
        return point_set,geod_sapmle,far_pts,knn_idx_12,knn_idx_21,point_set_grid,far_pts_point_3k      

    def pack_data(self,data):
        
        data_dicts ={}
        data_dicts['points'] = data[0]
        data_dicts['geod_sapmle'] = data[1]
        data_dicts['kpt'] = data[2]
        data_dicts['knn_idx_12'] = data[3]
        data_dicts['knn_idx_21'] = data[4]
        data_dicts['point_set_grid'] = data[5]
        data_dicts['far_pts_point_3k'] = data[6]
        return data_dicts

    def __getitem__(self, index):
        return self._get_item(index)
