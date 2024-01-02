from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from model import DGCNN_sample
from tqdm import tqdm
from dataload import DataLoader_set
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Out Dir
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml", type=str)
args = parser.parse_args()
params = OmegaConf.load(args.config)


savedir = params.savedir
outf = './models/' + savedir
writer = SummaryWriter('runs/'+savedir)
params.num_of_points = 4995
params.batch_size = 4
try:
    os.makedirs(outf)
except OSError:
    pass

trainset = DataLoader_set(params, 'train')
trainloader =torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True)
testset = DataLoader_set(params, 'test')
testloader =torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=False)



basis = DGCNN_sample(device, out=20)
checkpoint_path = outf + '/best_basis.pth'
checkpoint = torch.load(checkpoint_path)
basis.load_state_dict(checkpoint)
basis.to(device)

classifier = DGCNN_sample(device, out=40)
optimizer = torch.optim.Adam([{'params':classifier.parameters(), 'initial_lr': 0.002}], lr=0.002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001, last_epoch=600) 
classifier.to(device)

best_eval_loss = np.inf;

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    dist = dist.clamp(min=1e-12).sqrt() 
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = euclidean_dist(x, y)
    output = F.softmax(-alpha*distance, dim=-1)
    return output


def convert_C(C12, C21, Phi1, Phi2, alpha):
    T12 = knnsearch(torch.bmm(Phi1, C21), Phi2, alpha)
    T21 = knnsearch(torch.bmm(Phi2, C12), Phi1, alpha)

    return T12,T21

def bij_loss(pc_A, pc_B, phi_A, phi_B, G_A, G_B,D_A,D_B,phi_A_C,phi_B_C,device):
    p_inv_phi_A = torch.pinverse(phi_A)
    p_inv_phi_B = torch.pinverse(phi_B)
    c_G_A = torch.matmul(p_inv_phi_A, G_A)
    c_G_B = torch.matmul(p_inv_phi_B, G_B)
    c_G_At = torch.transpose(c_G_A,2,1)
    c_G_Bt = torch.transpose(c_G_B,2,1)

    C_BA = torch.matmul(c_G_A,torch.transpose(torch.pinverse(c_G_Bt),2,1))
    C_AB = torch.matmul(c_G_B,torch.transpose(torch.pinverse(c_G_At),2,1))

    T_BA, T_AB = convert_C(C_AB, C_BA, phi_A, phi_B, 30)

    Lisometric =  torch.mean(torch.square(torch.matmul(torch.matmul(T_AB , D_A) , torch.transpose(T_AB,1,2)) - D_B)) + torch.mean(torch.square(torch.matmul(torch.matmul(T_BA , D_B) , torch.transpose(T_BA,1,2)) - D_A)) 
    
    P_A = torch.matmul(T_BA,T_AB)
    P_B = torch.matmul(T_AB,T_BA)

    Lcyclic = torch.mean(torch.square( torch.matmul(torch.matmul(P_A , D_A) , torch.transpose(P_A,1,2) )- D_A)) + torch.mean(torch.square( torch.matmul(torch.matmul(P_B, D_B) , torch.transpose(P_B,1,2)) - D_B)) 
    return  Lcyclic + Lisometric 
# Training
for epoch in range(600):
    scheduler.step()
    train_loss = 0
    eval_loss = 0
    iters = 0
    train_c_loss = 0
    for batch in tqdm(trainloader, 0):
        batch_dicts = trainset.pack_data(batch)
        D = batch_dicts['geod_sapmle'].to(device)

        optimizer.zero_grad()
        classifier = classifier.train()
        with torch.no_grad():
            basis = basis.eval()
            pred = basis(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))       
            basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
            D_A = D[1:,:,:]; D_B = D[:-1,:,:]

        desc= classifier(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))        
        desc_A = desc[1:,:,:]; desc_B = desc[:-1,:,:]
        eucl_loss = bij_loss(None, None, basis_A, basis_B, desc_A, desc_B,D_A,D_B,None,None,device)

        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
        iters += 1
    writer.add_scalar('traindesc', train_loss / iters, global_step=epoch)
    iters = 0
    for batch in tqdm(testloader, 0):
        batch_dicts = trainset.pack_data(batch)
        D = batch['geod_sapmle'].to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            basis = basis.eval()
            pred = basis(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))        
            classifier = classifier.eval()
            desc = classifier(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))        
            basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
            desc_A = desc[1:,:,:]; desc_B = desc[:-1,:,:]
            D_A = D[1:,:,:]; D_B = D[:-1,:,:]

            eucl_loss = bij_loss(None, None, basis_A, basis_B, desc_A, desc_B,D_A,D_B,None,None,device)
            eval_loss +=   eucl_loss.item()
            iters += 1
    print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss/iters))
    writer.add_scalar('evaldesc', eval_loss / iters, global_step=epoch)
    if eval_loss <  best_eval_loss:
        print('save model')
        best_eval_loss = eval_loss
        torch.save(classifier.state_dict(), '%s/desc_best.pth' % (outf))
