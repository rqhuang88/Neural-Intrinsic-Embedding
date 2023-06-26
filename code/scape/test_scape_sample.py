import sys
sys.path.append('code/')
import torch
import numpy as np
import scipy.io
device = torch.device("cuda:0")
from dataload import SCAPE_5K_test_DataLoader as DataLoader
from model import DGCNN_sample


exp_dir = 'SCAPE_5k'
model_dir = 'SCAPE_5k'
savename = 'SCAPE_5k_test_on_scape_5k_sample_new'
DATA_PATH = 'data/SCAPE_5k/off'
TEST_DATASET = DataLoader(root=DATA_PATH,split='test')                                                
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0)

# Loading Models
basis_model = DGCNN_sample(device, out=20).to(device)
checkpoint = torch.load('models/'+ model_dir+'/best_basis.pth')
basis_model.load_state_dict(checkpoint)
basis_model = basis_model.eval()


classifier = DGCNN_sample(device, out=40).to(device)
checkpoint = torch.load('models/'+ model_dir+'/best_desc.pth')
classifier.load_state_dict(checkpoint)
classifier = classifier.eval()

data = []
FPS = []
Desc = []
with torch.no_grad():
    for datas in dataset_test:
        points = datas[0]
        fps = datas[1]
        points_grid = datas[2]
        points = points.transpose(2, 1)
        points = points.to(device)
        points_grid = points_grid.transpose(2, 1)
        points_grid = points_grid.to(device)
        pred = basis_model(points,points_grid,fps)
        desc = classifier(points,points_grid,fps)
        data.append(pred.squeeze().cpu().numpy())
        Desc.append(desc.squeeze().cpu().numpy())
        FPS.append(fps.squeeze().cpu().numpy())
        
data = np.array(data)
print(data.shape)
scipy.io.savemat('models/'+exp_dir+'/'+savename+'.mat',{'basis':data,'desc':Desc,'FPS':FPS})