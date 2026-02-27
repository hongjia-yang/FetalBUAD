import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dice import DiceLoss
from dataset import *
from unet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 8
ckpt_dir = './checkpoint'

def voxel_level_loss(output, target, t2_mask, noise_range=2.0):
    mask = t2_mask.float()  # [B, 1, D, H, W]
    target_expanded = target.view(-1, 1, 1, 1, 1).expand_as(output)
    if noise_range > 0:
        noise = torch.rand_like(output) * (2 * noise_range) - noise_range
        target_expanded = target_expanded* mask + noise * mask
    loss = F.l1_loss(output, target_expanded)
    return loss


#input data
dataset_train = FetalBrainAgeDataset(setname='train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=22,drop_last=True,pin_memory=True)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)

dataset_val = FetalBrainAgeDataset(setname='val')
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=22, drop_last=True,pin_memory=True)
num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)


#Model setting
num_epoch =150
st_epoch = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=unet().to(device)
torch.manual_seed(1)
optim = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,weight_decay=1e-6)
Segloss = DiceLoss(9)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer =optim,T_max=num_epoch)
min_val_loss=1000000
log=np.zeros([num_epoch,8])

#training
for epoch in range(st_epoch+1, num_epoch + 1):
    net.train()
    loss1_arr = []
    loss2_arr = []
    loss3_arr = []
    loss_arr = []
    for batch, data in enumerate(loader_train, 1):
        optim.zero_grad()
        image = data['image'].to(device)
        label = data['label'].to(device)
        seg = data['seg'].to(device)
        t2_mask = data['t2_mask'].to(device)
        age,output_seg,output_age = net(image)
        loss1=F.l1_loss(age,label)
        loss2,per_ch_score = Segloss(output_seg, seg)
        loss3=voxel_level_loss(output_age,label,t2_mask,2)
        print(loss1,loss2,loss3)
        loss=loss1+loss2*1000+loss3
        loss.backward()
        optim.step()
        loss_arr += [loss.item()]
        loss1_arr += [loss1.item()]
        loss2_arr += [loss2.item()]
        loss3_arr += [loss3.item()]
        print("TRAIN : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
    scheduler.step()
    log[epoch-1,0]=np.mean(loss_arr)
    log[epoch-1,1]=np.mean(loss1_arr)
    log[epoch-1,2]=np.mean(loss2_arr)
    log[epoch-1,3]=np.mean(loss3_arr)

    #validation
    with torch.no_grad():
        net.eval()
        loss1_arr = []
        loss2_arr = []
        loss3_arr = []
        loss_arr = []
        for batch, data in enumerate(loader_val, 1):
            image = data['image'].to(device)
            label = data['label'].to(device)
            seg = data['seg'].to(device)
            t2_mask = data['t2_mask'].to(device)
            age,output_seg,output_age = net(image)
            loss1=F.l1_loss(age,label)
            loss2,per_ch_score = Segloss(output_seg, seg)
            loss3=voxel_level_loss(output_age,label,t2_mask,2)
            print(loss1,loss2,loss3)
            loss=loss1+loss2*1000+loss3
            loss_arr += [loss.item()]
            loss1_arr += [loss1.item()]
            loss2_arr += [loss2.item()]
            loss3_arr += [loss3.item()]
            print("VALID : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" % (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
        if epoch>int(num_epoch/2) and np.mean(loss_arr)<min_val_loss:
            min_val_loss=np.mean(loss_arr)
            torch.save({'net': net.state_dict(), 'optim' : optim.state_dict()},"%s/MinValLoss.pth" % (ckpt_dir))
        torch.save({'net': net.state_dict(), 'optim' : optim.state_dict(), 'epoch':epoch, 'lr_schedule':scheduler.state_dict()},"%s/EachEpoch.pth" % (ckpt_dir))
    log[epoch-1,4]=np.mean(loss_arr)
    log[epoch-1,5]=np.mean(loss1_arr)
    log[epoch-1,6]=np.mean(loss2_arr)
    log[epoch-1,7]=np.mean(loss3_arr)
    np.save('log.npy',log)
