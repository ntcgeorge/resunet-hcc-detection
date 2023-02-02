from utils.Dataset import LiTDataset
from models.ResUnet import ResUnet
from metrics.Loss import PixelWiseDiceCE
from utils.util import display
from metrics.Metrics import SegAccuracy

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

import gc
import torch

from time import time
# hyperparameter
ALPHA = 0.33
LEARNING_RATE = 1e-4
EPOCH = 10
BATCH_SIZE = 12

# Define dataset and instanitiate DataLoader
ds = LiTDataset(augmentation=True)
train_ds, test_ds = random_split(ds, [0.8,0.2], generator=torch.Generator().manual_seed(55))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# release the memory cache
gc.collect()
torch.cuda.empty_cache()
#hyperparameter
alpha = 0.33
device = torch.device("cuda:0")
WEIGHTS = {"0":1, "1": 2, "2": 4}
EPOCH = 50
LEARNING_RATE = 1e-4

#instanitiate the net
net = ResUnet(1,3)
net = torch.nn.DataParallel(net).cuda()

# define loss function
loss_func = PixelWiseDiceCE(weights=torch.Tensor([1,3,9])) # set up weights for each label
loss_func = torch.nn.DataParallel(loss_func).cuda()

#define optimizer and learning_rate scheduler
opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [500, 750])

loss_history = []
val_loss_history = []

for epoch in range(EPOCH):
    epoch_start = time()
    lr_decay.step()

    print("epoch ", epoch)
    for step, (ct, seg) in enumerate(train_dl):
        step_start = time()
        ct = ct.to(device).float() # (B,1,H,W)
        seg = seg.type(torch.LongTensor)
        out = net(ct)
        loss = loss_func(out, seg)
        # record the loss
        loss_history.append(loss)
        # reset grad
        opt.zero_grad()
        # loss.requires_grad = True
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                acc = SegAccuracy(out, seg)
                dice = acc.cal_dice_coefficient()
                prec = acc.cal_precision()
                weighted_acc =acc.cal_weighted_acc(WEIGHTS)
                normal_acc = acc.cal_normal_acc()
                # jaccard = acc.cal_jaccard_index()
                print('step:{}, loss: {:.3f}, precision:{:.3f}, dice:{:.3f}, normal_acc:{:.3f}, weighted_acc:{:.3f}, step time:{:.3f} min'
                    .format(step, loss, prec, dice, normal_acc, weighted_acc, (time() - step_start) / 60))
                # display
                display(ct, out, seg)
                torch.save(net.state_dict(), './checkpoint/net{}-{:.3f}.pth'.format(epoch, loss))

    with torch.no_grad():
        # show the validation loss and accuracy
        v_loss_his = []
        for t_step, (t_ct, t_seg) in enumerate(test_dl):
            t_ct = t_ct.to(device).float()
            t_seg = t_seg.type(torch.uint8)
            t_out = net(t_ct)
            #instanitiate SegAccuracy Class
            #validation Loss
            val_loss = loss_func(t_out, t_seg)
            v_loss_his.append(val_loss)
            loss_his = torch.stack(v_loss_his)
            #validation accuracy
            #val_jaccard_index =  val_acc.cal_jaccard_index()
        mean_loss = loss_his.mean()
        val_loss_history.append(mean_loss)
        print('validation set loss: {:.3f}, epoch time:{:.2f} min'
            .format(mean_loss,(time() - epoch_start) / 60))