from __future__ import print_function
import sys
sys.path.append('code/')
import argparse
import os
import torch
import torch.optim as optim
from model import DGCNN_sample
from tqdm import tqdm
import numpy as np
from Loss import Get_loss
from dataload import DataLoader_set
from tqdm import tqdm
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

def train(params):
    if torch.cuda.is_available():
        device = torch.device(params.device)
    else:
        device = torch.device("cpu")

    if not os.path.exists('models/'+params.savedir):
        os.makedirs('models/'+params.savedir)
    save_name =  os.path.join('models/'+params.savedir,'params.txt')

    OmegaConf.save(params,save_name)
    print(device)
    writer = SummaryWriter('runs/'+params.savedir)
    # create model
    model = DGCNN_sample(device, out=params.n_feat).to(device)

    optimizer = torch.optim.Adam([{'params':model.parameters(), 'initial_lr': params.lr}], lr=params.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=params.lr/10, last_epoch=params.n_epochs) 
    # create dataset
    trainset = DataLoader_set(params, 'train')
    trainloader =torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True)
    testset = DataLoader_set(params, 'test')
    testloader =torch.utils.data.DataLoader(testset, batch_size=params.batch_size, shuffle=False)

    loss_function = Get_loss(params)
    # Training loop
    best_eval_loss = np.inf
    best_geo_loss = np.inf
    for epoch in range(1, params.n_epochs + 1):
        scheduler.step()
        iters = 0
        train_loss = 0
        model.train()
        train_loss_dict = {}
        for batch in tqdm(trainloader, 0):
            batch_dicts = trainset.pack_data(batch)

            # do iteration
            optimizer.zero_grad()
            pred = model(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))
            loss,loss_dict = loss_function(batch_dicts,pred,device)
            loss.backward()
            optimizer.step()

            # log and save model
            iters += 1
            for loss_name in loss_dict.keys():
                if loss_name not in train_loss_dict:
                    train_loss_dict[loss_name] = loss_dict[loss_name]
                else:
                    train_loss_dict[loss_name] += loss_dict[loss_name]
            train_loss += loss.item()
        for loss_name in train_loss_dict.keys():
            writer.add_scalar('train'+ loss_name, train_loss_dict[loss_name] / iters, global_step=epoch)
        
        
        with torch.no_grad():
            eval_loss_dict = {}
            eval_loss = 0    
            geo_loss = 0
            iters = 0
            model.eval()
            for batch in tqdm(testloader, 0):
                batch_dicts = testset.pack_data(batch)

                # do iteration
                optimizer.zero_grad()
                pred = model(batch_dicts['points'].transpose(2, 1).to(device),batch_dicts['point_set_grid'].transpose(2, 1).to(device),batch_dicts['far_pts_point_3k'].to(device))
                loss,loss_dict = loss_function(batch_dicts,pred,device)
            
                # log and save model
                iters += 1
                for loss_name in loss_dict.keys():
                    if loss_name not in eval_loss_dict:
                        eval_loss_dict[loss_name] = loss_dict[loss_name]
                    else:
                        eval_loss_dict[loss_name] += loss_dict[loss_name]
                eval_loss += loss.item()
                geo_loss += loss_dict['geo_loss']
            for loss_name in eval_loss_dict.keys():
                
                writer.add_scalar('test'+ loss_name, eval_loss_dict[loss_name]/ iters, global_step=epoch)
            

            if geo_loss <  best_geo_loss:

                print('save geo model at:'+str(epoch))
                best_geo_loss = geo_loss
                torch.save(model.state_dict(), '%s/%s' % ('models/'+params.savedir,'geo_'+params.save_ckpt_name)) 
            if eval_loss <  best_eval_loss:

                print('save model at:'+str(epoch))
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), '%s/%s' % ('models/'+params.savedir,params.save_ckpt_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    args = parser.parse_args()
    params = OmegaConf.load(args.config)

    print(params)
    train(params)
