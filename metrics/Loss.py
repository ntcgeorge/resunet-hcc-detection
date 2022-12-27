# Date: 2022-12-21
# Author: Lu Jiqiao, George
# Department : Polyu HTI
# ==============================================================================
'''The file provides a series of subclass of nn for loss fuction'''

import torch.nn as nn
import torch
from torchmetrics import Dice


class PixelWiseDiceCE(nn.Module):
    '''
    implement pixel-wise flatten crossentropy loss function.
    '''
    def __init__(self, weights=torch.Tensor([1,3,9])) -> None: # set up higher loss weight for label 1 and 2
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(weights) # mean loss
        self.weight = weights
        self.dice = Dice(average="micro", num_classes=3)

    def forward(self, output: torch.tensor, target: torch.tensor) -> torch.float32:
        '''
        Args:
            output: the output from the neural network with shape: (B, C, W, H)
            where C denotes the number of channel also the number of classes.

            target: the segmented image where every pixel enumerate in range(n_classes)
            with shape: (B, W, H)
        
        Return: Loss value in float32
        '''
        batch_size = output.shape[0]
        ce_loss = 0.
        for i in range(batch_size):
            label = target[i].squeeze(0)
            x = output[i]
            # flatten target and x
            label = label.flatten()
            x = x.flatten(start_dim=1).T #need to transpose to adapt to the shape of target
            ce_loss += self.loss_func(x, label)

        ce_loss = ce_loss / batch_size
        dice_coe = self.dice(output, target)
        return ce_loss + (1 - dice_coe)