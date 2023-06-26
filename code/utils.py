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

    
def VF_adjacency_matrix(V, F):
    """
    Input:
    V: N x 3
    F: F x 3
    Outputs:
    C: V x F adjacency matrix
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    VF_adj = torch.zeros((V.shape[0], F.shape[0]), dtype=dtype, device=device)
    v_idx = F.view(-1)
    f_idx = torch.arange(F.shape[0]).repeat(3).reshape(3, F.shape[0]).transpose(1, 0).contiguous().view(
        -1)  # [000111...FFF]

    VF_adj[v_idx, f_idx] = 1
    return VF_adj




# def _grad_div(V,T):
    
#     #tensor type and device
#     device = V.device
#     dtype = V.dtype
    
#     #WARNING not sure about this
#     V = V.reshape([-1,3])
#     T = T.reshape([-1,3])

#     XF = V[T,:].transpose(0,1)

#     Na = torch.cross(XF[1]-XF[0],XF[2]-XF[0])
#     # print(Na[:10])
#     A = torch.sqrt(torch.sum(Na**2,-1,keepdim=True))#+1e-6
#     # print(A[:10])
#     N = Na/A
#     dA = 0.5/A

#     m = T.shape[0]
#     n = V.shape[0]
#     def grad(f):
#         gf = torch.zeros(m,3,f.shape[-1], device=device, dtype=dtype)
#         for i in range(3):
#             s = (i+1)%3
#             t = (i+2)%3
#             v = -torch.cross(XF[t]-XF[s],N)
#             if SAVE_MEMORY:
#                 gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) #Slower less-memeory
#             else:
#                 gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) 
#         return gf
    
#     def div(f):
#         gf = torch.zeros(f.shape[-1],n, device=device, dtype=dtype)        
#         for i in range(3):
#             s = (i+1)%3
#             t = (i+2)%3
#             v = torch.cross(XF[t]-XF[s],N)
#             if SAVE_MEMORY:
#                 gf.add_(scatter_add( torch.bmm(v[:,None,:],f)[:,0,:].t(), T[:,i], dim_size=n))# slower but uses less memory
#             else:
#                 gf.add_(scatter_add( (f*v[:,:,None]).sum(1).t(), T[:,i], dim_size=n))
#         return gf.t()
    

#     return grad, div, A





def grad_loss(f,V,T):
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    #WARNING not sure about this
    V = V.reshape([-1,3])
    T = T.reshape([-1,3])

    XF = V[T,:].transpose(0,1)

    Na = torch.cross(XF[1]-XF[0],XF[2]-XF[0])
    A = torch.sqrt(torch.sum(Na**2,-1,keepdim=True))#+1e-6
    N = Na/A
    dA = 0.5/A

    m = T.shape[0]
    
    gf = torch.zeros(m,3,f.shape[-1], device=device, dtype=dtype)
    for i in range(3):
        s = (i+1)%3
        t = (i+2)%3
        v = -torch.cross(XF[t]-XF[s],N)
        gf.add_(f[T[:,i],None,:]*(dA[:,0,None,None]*v[:,:,None])) 
    
    return gf





def distance_GIH(V, T, t=1e-1):
    
    W,A = LBO_slim(V, T)
    
    grad,div,N = _grad_div(V,T)
    
    D = _geodesics_in_heat(grad,div,W[0],A,t)
    d = torch.diag(D)[:,None]
    #WARNIG: original D is not symmetric, it is symmetrized and shifted to have diagonal equal to zero
    D = (D + D.t()-d-d.t())/2

    return D, grad, div, W, A, N




def LBO_slim(V, F):
    """
    Input:
      V: B x N x 3
      F: B x F x 3
    Outputs:
      C: B x F x 3 list of cotangents corresponding
        angles for triangles, columns correspond to edges 23,31,12
    """
    #tensor type and device
    device = V.device
    dtype = V.dtype
    
    indices_repeat = torch.stack([F, F, F], dim=2)

    # v1 is the list of first triangles B*F*3, v2 second and v3 third
    v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
    v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
    v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())

    l1 = torch.sqrt(((v2 - v3) ** 2).sum(2))  # distance of edge 2-3 for every face B*F
    l2 = torch.sqrt(((v3 - v1) ** 2).sum(2))
    l3 = torch.sqrt(((v1 - v2) ** 2).sum(2))


    # Heron's formula for area
    A = 0.5 * (torch.sum(torch.cross(v2 - v1, v3 - v2, dim=2) ** 2, dim=2) ** 0.5)  # VALIDATED

    # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
    cot23 = (l1 ** 2 - l2 ** 2 - l3 ** 2) / (4 * A)
    cot31 = (l2 ** 2 - l3 ** 2 - l1 ** 2) / (4 * A)
    cot12 = (l3 ** 2 - l1 ** 2 - l2 ** 2) / (4 * A)
    batch_cot23 = cot23.view(-1)
    batch_cot31 = cot31.view(-1)
    batch_cot12 = cot12.view(-1)


    B = V.shape[0]
    num_vertices_full = V.shape[1]
    num_faces = F.shape[1]

    edges_23 = F[:, :, [1, 2]]
    edges_31 = F[:, :, [2, 0]]
    edges_12 = F[:, :, [0, 1]]

    batch_edges_23 = edges_23.view(-1, 2)
    batch_edges_31 = edges_31.view(-1, 2)
    batch_edges_12 = edges_12.view(-1, 2)

    W = torch.zeros(B, num_vertices_full, num_vertices_full, dtype=dtype, device=device)

    repeated_batch_idx_f = torch.arange(0, B).repeat(num_faces).reshape(num_faces, B).transpose(1, 0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_faces
    repeated_batch_idx_v = torch.arange(0, B).repeat(num_vertices_full).reshape(num_vertices_full, B).transpose(1,
                                                                                                      0).contiguous().view(
        -1)  # [000...111...BBB...], number of repetitions is: num_vertices_full
    repeated_vertex_idx_b = torch.arange(0, num_vertices_full).repeat(B)

    W[repeated_batch_idx_f, batch_edges_23[:, 0], batch_edges_23[:, 1]] = batch_cot23
    W[repeated_batch_idx_f, batch_edges_31[:, 0], batch_edges_31[:, 1]] = batch_cot31
    W[repeated_batch_idx_f, batch_edges_12[:, 0], batch_edges_12[:, 1]] = batch_cot12

    W = W + W.transpose(2, 1)

    batch_rows_sum_W = torch.sum(W, dim=1).view(-1)
    W[repeated_batch_idx_v, repeated_vertex_idx_b, repeated_vertex_idx_b] = -batch_rows_sum_W


    VF_adj = VF_adjacency_matrix(V[0], F[0]).unsqueeze(0).expand(B, num_vertices_full, num_faces)  # VALIDATED
    V_area = (torch.bmm(VF_adj, A.unsqueeze(2)) ).squeeze()      
    return W, V_area
    





def data_augmentation(point_set):
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        #point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        return point_set


def load_mesh(path, torch_tensors=False):
    VERT = np.loadtxt(path+'/mesh.vert')
    TRIV = np.loadtxt(path+'/mesh.triv',dtype='int32')-1
    if torch_tensors:
        VERT = torch.from_numpy(VERT)
        TRIV = torch.from_numpy(TRIV)
    
    return VERT, TRIV
def pc_normalize(pc):
    
    centroid = torch.mean(pc, axis=0)

    pc = pc - centroid
    #m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    #pc = pc / m
    return pc
import math

def create_rotation_matrix_z():
    alpha = torch.rand(1) * 360
    alpha = alpha / 180 * math.pi
    cosval = torch.cos(alpha)
    sinval = torch.sin(alpha)
    rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return rotation_matrix

def create_rotation_matrix(axis = 1):
    alpha = torch.rand(1) * 360
    alpha = alpha / 180 * math.pi
    c = torch.cos(alpha)
    s = torch.sin(alpha)
    rot_2d = torch.as_tensor([[c, -s], [s, c]], dtype=torch.float)
    rot_3d = torch.eye(3, dtype=torch.float32)
    idx = [i for i in range(3) if i != axis]
    for i in range(len(idx)):
        for j in range(len(idx)):
            rot_3d[idx[i], idx[j]] = rot_2d[i, j]
    return rot_3d


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




def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k]
        rotated_data[k,:,0:3] = np.dot(shape_pc, rotation_matrix)
    return rotated_data, rotation_matrix

def rotate_point_cloud_by_angle_a(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0], 
                                    [0, 0, 1]])
        shape_pc = batch_data[k]
        rotated_data[k] = np.dot(shape_pc, rotation_matrix)
    #print("a",np.sum(rotated_data.dot(rotation_matrix.T)-batch_data))  
    return rotated_data,rotation_matrix


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# """
# This module implements utility functions for sampling points from
# batches of meshes.
# """
# import sys
# from typing import Tuple, Union


# from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
# from pytorch3d.ops.packed_to_padded import packed_to_padded
# from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments


# def sample_points_from_meshes(
#     meshes,
#     num_samples: int = 10000,
#     return_normals: bool = False,
#     return_textures: bool = False,
# ) -> Union[
#     torch.Tensor,
#     Tuple[torch.Tensor, torch.Tensor],
#     Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
# ]:
#     """
#     Convert a batch of meshes to a batch of pointclouds by uniformly sampling
#     points on the surface of the mesh with probability proportional to the
#     face area.

#     Args:
#         meshes: A Meshes object with a batch of N meshes.
#         num_samples: Integer giving the number of point samples per mesh.
#         return_normals: If True, return normals for the sampled points.
#         return_textures: If True, return textures for the sampled points.

#     Returns:
#         3-element tuple containing

#         - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
#           coordinates of sampled points for each mesh in the batch. For empty
#           meshes the corresponding row in the samples array will be filled with 0.
#         - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
#           to each sampled point. Only returned if return_normals is True.
#           For empty meshes the corresponding row in the normals array will
#           be filled with 0.
#         - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
#           texture vector to each sampled point. Only returned if return_textures is True.
#           For empty meshes the corresponding row in the textures array will
#           be filled with 0.

#         Note that in a future releases, we will replace the 3-element tuple output
#         with a `Pointclouds` datastructure, as follows

#         .. code-block:: python

#             Pointclouds(samples, normals=normals, features=textures)
#     """
#     if meshes.isempty():
#         raise ValueError("Meshes are empty.")

#     verts = meshes.verts_packed()
#     if not torch.isfinite(verts).all():
#         raise ValueError("Meshes contain nan or inf.")

#     faces = meshes.faces_packed()
#     mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
#     num_meshes = len(meshes)
#     num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

#     # Initialize samples tensor with fill value 0 for empty meshes.
#     samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

#     # Only compute samples for non empty meshes
#     with torch.no_grad():
#         areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
#         max_faces = meshes.num_faces_per_mesh().max().item()
#         areas_padded = packed_to_padded(
#             areas, mesh_to_face[meshes.valid], max_faces
#         )  # (N, F)

#         # TODO (gkioxari) Confirm multinomial bug is not present with real data.
#         sample_face_idxs = areas_padded.multinomial(
#             num_samples, replacement=True
#         )  # (N, num_samples)
#         sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

#     # Get the vertex coordinates of the sampled faces.
#     face_verts = verts[faces]
#     v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

#     # Randomly generate barycentric coords.
#     w0, w1, w2 = _rand_barycentric_coords(
#         num_valid_meshes, num_samples, verts.dtype, verts.device
#     )
#     # Use the barycentric coords to get a point on each sampled face.
#     a = v0[sample_face_idxs]  # (N, num_samples, 3)
#     b = v1[sample_face_idxs]
#     c = v2[sample_face_idxs]
#     samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c


#     return samples 



# def _rand_barycentric_coords(
#     size1, size2, dtype: torch.dtype, device: torch.device
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Helper function to generate random barycentric coordinates which are uniformly
#     distributed over a triangle.

#     Args:
#         size1, size2: The number of coordinates generated will be size1*size2.
#                       Output tensors will each be of shape (size1, size2).
#         dtype: Datatype to generate.
#         device: A torch.device object on which the outputs will be allocated.

#     Returns:
#         w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
#             coordinates
#     """
#     uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
#     u, v = uv[0], uv[1]
#     u_sqrt = u.sqrt()
#     w0 = 1.0 - u_sqrt
#     w1 = u_sqrt * (1.0 - v)
#     w2 = u_sqrt * v
#     # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
#     #  got `Tuple[float, typing.Any, typing.Any]`.
#     return w0, w1, w2