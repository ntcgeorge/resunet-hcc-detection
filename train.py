from utils.Dataset import LiTDataset
from models.ResUnet import ResUnet
from metrics.Loss import PixelWiseCE
from utils.util import display

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from time import time
# hyperparameter
ALPHA = 0.33
LEARNING_RATE = 1e-4
EPOCH = 10


ds = LiTDataset()
train_ds, test_ds, val_ds = random_split(ds, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(ds, batch_size=48, shuffle=True)
net = ResUnet(1,3)
net = torch.nn.DataParallel(net).cuda()
loss_func = PixelWiseCE()
loss_func = torch.nn.DataParallel(loss_func).cuda()
opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [500, 750])

# define dataset
ds = LiTDataset(augmentation=True)
train_ds, test_ds, val_ds = random_split(ds, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))
device = torch.device("cuda:0")
start = time()
loss_history = []
for epoch in range(10):
  lr_decay.step()

  for step, (ct, seg) in enumerate(train_dl):
    ct = ct.to(device).float()
    seg = seg.type(torch.LongTensor)
    out = net(ct)
    loss = loss_func(out, seg)
    loss_history.append(loss)
    
    # reset grad
    opt.zero_grad()
    # loss.requires_grad = True
    loss.backward()
    opt.step()
    if step % 100 == 0:
      x = ct.to('cpu').detach().numpy()
      y = out.to('cpu').detach().numpy()
      print("number of prediction: ", np.unique(y).shape[0])
      display(x[-1], np.argmax(y[-1],0))
      print('epoch:{}, step:{}, loss: {:.3f}, time:{:.3f} min'
            .format(epoch, step, loss, (time() - start) / 60))
      torch.save(net.state_dict(), './checkpoint/net{}-{:.3f}.pth'.format(epoch, loss))