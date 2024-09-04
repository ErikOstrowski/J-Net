import numpy as np

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
from dataset_32 import *
from JNet_32 import UNet
from utils import *
from config_32 import *



from torch.nn.functional import cross_entropy


cfg = Config()
train_transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    RandomFlip(),
    ToTensor(),
])
val_transform = transforms.Compose([
    GrayscaleNormalization(mean=0.5, std=0.5),
    ToTensor(),
])

pred_img = './pred_out/'

if not os.path.exists(pred_img):
	os.makedirs(pred_img)





# Set Dataset
train_dataset = Dataset(imgs_dir=TRAIN_IMGS_DIR, labels_dir=TRAIN_LABELS_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_dataset = Dataset(imgs_dir=VAL_IMGS_DIR, labels_dir=VAL_LABELS_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

train_data_num = len(train_dataset)
val_data_num = len(val_dataset)

train_batch_num = int(np.ceil(train_data_num / cfg.BATCH_SIZE))
val_batch_num = int(np.ceil(val_data_num / cfg.BATCH_SIZE))

# Network
net = UNet(1,1,16).to(device)
# Loss Function
loss_fn = nn.BCEWithLogitsLoss().to(device)
loss_l1 = nn.L1Loss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

# Tensorboard
train_writer = SummaryWriter(log_dir=TRAIN_LOG_DIR)
val_writer = SummaryWriter(log_dir=VAL_LOG_DIR)

# Training
start_epoch = 0

#Load Checkpoint File
#if os.listdir(CKPT_DIR):
#    net, optim, start_epoch = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)
#else:
#    print('* Training from scratch')



        

num_epochs = cfg.NUM_EPOCHS
for epoch in range(start_epoch,num_epochs):
    net.train()  # Train Mode
    train_loss_arr = list()
    
    for batch_idx, data in enumerate(train_loader, 1):
        # Forward Propagation
        img = data['img_d'].to(device)
        label = data['label'].to(device)
        img = img.float()
        label_d = data['label_d'].to(device)
        
        

        

        id = data['id']

        output1, output2, output3, output4 = net(img.float())

        
        
        # Backward Propagation
        

        
        optim.zero_grad()
        nplable = np.array(label)
        

         
        
        #lable2 = np.zeros([cfg.BATCH_SIZE,1,32, 32])
        lable3 = np.zeros([cfg.BATCH_SIZE,1,64, 64])
        lable4 = np.zeros([cfg.BATCH_SIZE,1,128, 128])
        #lable2 = np.zeros([cfg.BATCH_SIZE,1,160, 160])

        
        
        
        
        
        for b in range(cfg.BATCH_SIZE):
          #lable2[b][0] = cv2.resize(nplable[b][0], ( 160, 160 ), interpolation= cv2.INTER_LINEAR)
          
          #lable2[b][0] = cv2.resize(nplable[b][0], ( 32, 32 ), interpolation= cv2.INTER_LINEAR)
        
          lable3[b][0] = cv2.resize(nplable[b][0], ( 64, 64 ), interpolation= cv2.INTER_LINEAR)
          
          lable4[b][0] = cv2.resize(nplable[b][0], ( 128, 128 ), interpolation= cv2.INTER_LINEAR)
          
          
        #lable2 = torch.tensor(lable2)
        lable3 = torch.tensor(lable3)
        lable4 = torch.tensor(lable4)



        
        loss1 = loss_fn(output1, label_d)
        loss2 = loss_fn(output2,lable3)
        loss3 = loss_fn(output3,lable4)
        loss4 = loss_fn(output4, label)
        #loss5 = loss_fn(output5, label)
       
        
      
        loss = loss1 + loss3 + loss4 + loss2
        loss.backward()
        
        optim.step()
        
        # Calc Loss Function
        train_loss_arr.append(loss.item())
        print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss_arr[-1]))
        
                
        global_step = train_batch_num * (epoch-1) + batch_idx
        
    train_loss_avg = np.mean(train_loss_arr)
    train_writer.add_scalar(tag='loss', scalar_value=train_loss_avg, global_step=epoch)
    
    # Validation (No Back Propagation)
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        val_loss_arr = list()
        
        for batch_idx, data in enumerate(val_loader, 1):
            # Forward Propagation
            img = data['img_d'].to(device)
            label = data['label'].to(device)
            img = img.float()
            id = data['id']
            
            
            output1, output2, output3, output = net(img)
            

            if epoch ==num_epochs-1:
             for b in range(cfg.BATCH_SIZE):
             

              np.save(pred_img + id[b],output[b].cpu())
            
            # Calc Loss Function
            loss = loss_fn(output, label)
            val_loss_arr.append(loss.item())
            
            print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss_arr[-1]))
            
             
            global_step = val_batch_num * (epoch-1) + batch_idx
            
    val_loss_avg = np.mean(val_loss_arr)
    val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)
    
    print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f}'
    print(print_form.format(epoch, train_loss_avg, val_loss_avg))
    if epoch % 10 == 0:
     save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)
    if epoch == 4:
     save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)
    if epoch == 1:
     save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)
    
train_writer.close()
val_writer.close()
