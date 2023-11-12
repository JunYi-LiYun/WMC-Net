import time
import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.pretrain_datasets import TrainData, ValData
from models.FFA import FFANet
# from models.MSBDN import MSBDNNet
from models.MSBDNRDFFT3 import MSBDNNet
# from models.qrc0 import Dehaze
from models.MSBDNRDFFTransformer import MSBDNNet
from utils import to_psnr, print_log, validation, adjust_learning_rate
from utils import *
from torchvision.models import vgg16
import math
from pdb import set_trace as bp
# parser = argparse.ArgumentParser()
# parser.add_argument('--backbone', type=str, default='MSBDNNet', help='Backbone model(GCANet/FFANet/MSBDNNet)')
# parser.add_argument('--pretrain_model_dir', type=str, default='output1/')
# opt = parser.parse_known_args()[0]
# #from perceptual import LossNetwork

def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr

    
lr=1e-4
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
crop_size = [256, 256]
train_batch_size = 2
val_batch_size = 1
num_epochs = 2000
category = 'outdoor'

#img_dir = 'D:/Pycharm/GridDehazeNet-master/realtest/'
#output_dir = 'D:/Pycharm/GridDehazeNet-master/unlabeled/clear_1/'
val_data_dir = 'O-HAZE/FD/'
train_data_dir = 'O-HAZE/# O-HAZY NTIRE 2018/'
gps=3
blocks=19
# net = FFANet(gps, blocks)
# net = MSBDNNet()
net = MSBDNNet()
# net = Dehaze()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
# net.load_state_dict(torch.load('D:/datasethaze/out//haze_current148'))
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=0)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=0)
print("DATALOADER DONE!")
#old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
#print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
torch.backends.cudnn.benchmark = True
old_val_psnr = 0
all_T = 38000
for epoch in range(num_epochs):
    psnr_list = []
    start_time = time.time()
    #adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):
        if batch_id >19:
            break
        step_num = batch_id + epoch * 19 + 1
        lr=lr_schedule_cosdecay(step_num,all_T)           # 调整学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)
        #dc = get_dark_channel(haze, 15)
        #A = get_atmosphere(haze, dc, 0.0001)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        # _, J, T, A, I = net(haze)        # 输入雾图进入预训练网络
        J= net(haze)
        #s, v = get_SV_from_HSV(J)
        #CAP_loss = F.smooth_l1_loss(s, v)
        Rec_Loss1 = F.smooth_l1_loss(J, gt)    # 原始网络训练损失
        # Rec_Loss2 = F.smooth_l1_loss(I, haze)  # 重建损失

        #perceptual_loss = loss_network(dehaze, gt)
        # loss = Rec_Loss1 + Rec_Loss2          # 预训练损失
        loss = Rec_Loss1 # 预训练损失

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(J, gt))

        #if not (batch_id % 100):
        # print('Epoch: {}, Iteration: {}, Loss: {}, Rec_Loss1: {}, Rec_loss2: {}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2))
        print('Epoch: {}, Iteration: {}, Loss: {}, Rec_Loss1: {}'.format(epoch, batch_id, loss, Rec_Loss1))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    if epoch % 100 == 0:
        torch.save(net.state_dict(), 'out/haze_current{}'.format(epoch))

    # --- Use the evaluation model in testing --- #

     # net = load_model(opt.backbone, opt.pretrain_model_dir, device, device_ids)
    net.eval()


    val_psnr, val_ssim = validation(net,MSBDNNet, val_data_loader, category='outdoor')
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)
    # print('Val_PSNR:{0:.2f}, Val_SSIM:{1:.4f}'.format(val_psnr ,val_ssim))
    # --- update the network weight --- #
    #if val_psnr >= old_val_psnr:
    #    torch.save(net.state_dict(), '{}_haze_best_{}_{}'.format(category, network_height, network_width))
    #    old_val_psnr = val_psnr