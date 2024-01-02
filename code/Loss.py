
from __future__ import print_function
import torch
import torch.nn as nn

def Get_loss(params):

    if params.Loss == 'BIJ_KL_loss':
        return BIJ_KL_loss

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)


def BIJ_KL_loss(data_dicts, pred, device):
    KLloss = torch.nn.KLDivLoss()
    geo_kpt = data_dicts['geod_sapmle'] .to(device)

    kpt = data_dicts['kpt'].to(device).squeeze(1)

    knn_idx_21 = data_dicts['knn_idx_21'].to(device).squeeze(-1)
    knn_idx_12 = data_dicts['knn_idx_12'].to(device).squeeze(-1)

    B, N, C = pred.shape
    N_kpt = kpt.shape[-1]
    idx_base = torch.arange(0, B).view(B,-1).to(device) * N
    kpt = kpt + idx_base
    kpt = kpt.view(-1)
    Basis_kpt = pred.view(B*N, C)[kpt].view(B, N_kpt, 1, C) # B N_kpt N C
    Eucli_distance_kpt = torch.norm(pred.view(B,1,N,C) - Basis_kpt,p=2,dim = 3) 

    distance_Weight = geo_kpt.clone()
    distance_Weight[distance_Weight == 0] = 1
    distance_Weight = 1 / distance_Weight**2

    geo_kpt_loss_gt = torch.sqrt(torch.mean(torch.square(Eucli_distance_kpt - geo_kpt) * distance_Weight))
    


    s_max = torch.nn.Softmax(dim=-1)
    pred_input = torch.log(s_max(-10*Eucli_distance_kpt))
    gt_target = s_max(-10*geo_kpt)

    kl_loss = KLloss(pred_input,gt_target)

    B1 = pred.view(B*N,C)[kpt].view(B,N_kpt,C)[:,0::2,:] 
    B2 = pred.view(B*N,C)[kpt].view(B,N_kpt,C)[:,1::2,:] 
    pseudo_inv_2 = torch.pinverse(B2)
    pseudo_inv_1 = torch.pinverse(B1)

    B1_tranfer = []
    for b_idx in range(B):  
        B1_tranfer.append(B1[b_idx,knn_idx_21[b_idx],:])
    B1_tranfer = torch.stack(B1_tranfer,dim=0)
    B2_tranfer = []
    for b_idx in range(B):  
        B2_tranfer.append(B2[b_idx,knn_idx_12[b_idx],:])
    B2_tranfer = torch.stack(B2_tranfer,dim=0)
    frob_loss = FrobeniusLoss()
    C_12 = torch.matmul(pseudo_inv_2,B1_tranfer) # knn_idx  P2 -> P1
    C_21 = torch.matmul(pseudo_inv_1,B2_tranfer) # knn_idx  P1 -> P2
    I = torch.eye(C, device=device).repeat(B, 1, 1)
    bij_loss = torch.mean(frob_loss(torch.bmm(C_12, C_21), I) + frob_loss(torch.bmm(C_21, C_12), I)) * 0.5

    eucl_loss = bij_loss
    loss_dict = {'geo_loss':geo_kpt_loss_gt.item(),'eucl_loss':eucl_loss.item(),'kl_loss':kl_loss.item()}

    return  geo_kpt_loss_gt +  eucl_loss + kl_loss, loss_dict


 


