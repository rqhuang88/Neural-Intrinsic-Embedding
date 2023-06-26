from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv42 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.dense1 = torch.nn.Linear(1024,256)
        self.dense2 = torch.nn.Linear(256,256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(128)
        self.bn42 = nn.BatchNorm1d(128)  
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat
class PointNetBasis(nn.Module):
    def __init__(self, k = 20, feature_transform=False):
        super(PointNetBasis, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv2b = torch.nn.Conv1d(256, 256, 1)
        self.conv2c = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2b = nn.BatchNorm1d(256)
        self.bn2c = nn.BatchNorm1d(256)
        # self.m   = nn.Dropout(p=0.3)
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = F.relu(self.bn2c(self.conv2c(x)))
        # x = self.m(x)
        #x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv3(x)
        x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x



############ DESC NETWORK

class PointNetfeatDesc(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeatDesc, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv42 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.dense1 = torch.nn.Linear(1024,256)
        self.dense2 = torch.nn.Linear(256,256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn41 = nn.BatchNorm1d(128)
        self.bn42 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(256)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn6(self.dense1(x)))
        x = F.relu(self.bn7(self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDesc(nn.Module):
    def __init__(self, k = 40, feature_transform=False):
        super(PointNetDesc, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeatDesc(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.m   = nn.Dropout(p=0.3)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.m(x)
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x#, trans, trans_feat




class DGCNN_sample(nn.Module):
    def __init__(self, device, out = 128,emb_dims = 512):
        super(DGCNN_sample, self).__init__()
        self.k = 50
        self.emb_dims = emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(128)
        self.bn8 = nn.BatchNorm1d(64)
        self.device = device
        self.out = out
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(192+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv8 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                        self.bn8,
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Conv1d(64, self.out, kernel_size=1, bias=False)
                

    def forward(self, x,x_grid,FPS):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature_sample(x, x_grid, k=self.k,device=self.device )
        x = self.conv1(x)                       
        x = self.conv2(x)                       
        x1 = x.max(dim=-1, keepdim=False)[0]    
        x1_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x1_grid = torch.cat([x1_grid,x1[i,:,FPS[i]].unsqueeze(0)],dim=0)
            
        x = get_graph_feature_sample(x1, x1_grid,k=self.k,device=self.device)     
        x = self.conv3(x)                       
        x = self.conv4(x)                       
        x2 = x.max(dim=-1, keepdim=False)[0]   
        x2_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x2_grid = torch.cat([x2_grid,x2[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x = get_graph_feature_sample(x2,x2_grid, k=self.k,device=self.device)     
        x = self.conv5(x)                       
        x3 = x.max(dim=-1, keepdim=False)[0]    
        x3_grid = torch.tensor([],device=self.device)
        for i in range(batch_size):
            x3_grid = torch.cat([x3_grid,x3[i,:,FPS[i]].unsqueeze(0)],dim=0)
        x_grid = torch.cat((x1_grid, x2_grid, x3_grid), dim=1)      
        x_grid = self.conv6(x_grid)                       
        x_grid = x_grid.max(dim=-1, keepdim=True)[0]       
        
        x_grid = x_grid.repeat(1, 1, num_points)        
        x = torch.cat((x_grid, x1, x2, x3), dim=1)   
        x = self.conv7(x)                      
        x = self.conv8(x)                      
        x = self.conv9(x)                       
        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        return x

def get_graph_feature_sample(x,x_grid, k=20, idx=None, dim9=False,device='cuda'):
    batch_size = x.size(0)
    num_points = x.size(2)
    num_points_sample = x_grid.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knnsearch(x_grid.transpose(1,2),x.transpose(1,2), k=1)   # (batch_size, num_points, k)
            idx_grid = knn(x_grid, k=k)
            idx_final = torch.tensor([],device=x.device).long()
            for i in range(batch_size):
                idx_grid_batch = idx_grid[i]
                idx_batch = idx[i].squeeze()
                idx_final = torch.cat((idx_final, idx_grid_batch[idx_batch,:].unsqueeze(0)), dim=0)
        else:
            idx = knn(x[:, 6:], k=k)
    idx = idx_final
    device = device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points_sample

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   
    x_grid = x_grid.transpose(2, 1).contiguous()
    feature = x_grid.view(batch_size*num_points_sample, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    for i in range(batch_size):
        feature[i,:,0,:] = x[i]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature     

def knnsearch(tar, src, k):
    bat_size = tar.shape[0]
    idx = torch.tensor([],device=tar.device).long()
    for i in range(bat_size):
        P1 = src[i]
        P2 = tar[i]
        pairwise_distance = -  torch.cdist(P1,P2)
        _ , t_st= pairwise_distance.topk(k=k, dim=-1)

        idx = torch.cat((idx, t_st.unsqueeze(0)), dim=0)
    return idx

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx   