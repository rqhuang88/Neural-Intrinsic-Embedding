from cv2 import norm
import torch
import torch.nn.functional as F
import numpy as np

import time



SAVE_MEMORY = False
def knnsearch(tar, src, k):
    P1 = src
    P2 = tar
    N1,C = P1.shape
    N2,C = P2.shape

    pairwise_distance = -  torch.sqrt(torch.sum(torch.square(P1.view(N1,1,C) - P2.view(1,N2,C)),dim=-1))
    dist , T_st = pairwise_distance.topk(k=k, dim=-1)# (batch_size, num_points, k)
    return T_st, dist


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

    
def pc_normalize(pc):
    
    centroid = torch.mean(pc, axis=0)

    pc = pc - centroid
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)                                          
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # point = point[centroids.astype(np.int32)]

    return centroids

def farthest_point_sample_torch(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point
    centroids = torch.zeros((npoint,)).float().cuda()
    distance = torch.ones((N,)).float().cuda() * 1e10
    farthest = torch.randint(0, N,[1]).long().cuda()                 
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    # point = point[centroids.astype(np.int32)]

    return centroids

