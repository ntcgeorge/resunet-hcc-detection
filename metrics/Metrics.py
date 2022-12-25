import torch
import numpy as np
from typing import *
from torchmetrics import JaccardIndex, Precision, Dice

class SegAccuracy():
    def __init__(self, pred: torch.Tensor, truth: torch.Tensor) -> None:
        '''
        The class provides a series of metrics for measuring the accuracy of
        segmentation.

        Args
            pred: the output from network, the shape should be (B, C, H, W), where
            the C is the number of classes.

            truth: the ground truth of the prediction, the shape is (B, 1, H, W) whose pixel
            enumerating in the range(classes_num) represent the label of the local pixel
        '''
        self.pred = pred.cpu()
        # print("pred dtype is: ", self.pred.dtype, "shape: ", self.pred.shape)
        self.truth = truth.cpu()
        # print("truth dtype is: ", self.truth.dtype, "shape: ", self.truth.shape)
        self.pred_argmax = torch.argmax(self.pred, 1, keepdim=True) # (B, 1, H, W)
        
    def cal_normal_acc(self) -> float:
        '''
        Calculate the normal accuracy by measuring the overlapping rate with ground truth.

        Return the normal accuracy.
        '''
        acc = (self.pred_argmax == self.truth).type(torch.float32).mean().item()
        assert acc <= 1 , f"accuracy {acc} excess 1!"
        assert  acc >= 0, f"accuracy {acc} is negative!"
        return acc
    
    def cal_weighted_acc(self, weights) -> float:
        '''
        Calculate weighted accuracy, the accuracy weights differently as weights parameter
        indicates. The larger the weight, the more accurate is for a single correctly classified label.

        (weight * (pred == truth)).sum() / (weight * torch.nonzero(truth)).sum()

        Arg
            weights: weights indicats the significance of different labels
        '''
        #unpack the dict
        w_0 = weights["0"]
        w_1 = weights["1"]
        w_2 = weights["2"]
        
        w = [w_0, w_1, w_2]
        w_mat = self.truth.apply_(lambda x : w[int(x)])
        oh_truth = torch.where(self.truth > 0, 1, 0)
        pred_weight = w_mat * (self.pred_argmax == self.truth).type(torch.float32)
        pred_sum = pred_weight.sum().item()
        truth_weight = w_mat * oh_truth
        truth_sum = truth_weight.sum().item()
        acc = pred_sum / truth_sum
        assert acc <= 1 , f"accuracy {acc} excess 1!"
        assert  acc >= 0, f"accuracy {acc} is negative!"
        return acc
    
    def cal_dice_coefficient(self) -> float:
        '''
        Return
            dice coefficient, the index is based on the overlapping area to union area ratio.
        '''
        intersection = (self.pred_argmax == self.truth).type(torch.float32).sum()
        union = (self.pred_argmax != 0).sum() + (self.truth != 0).sum()
        
        return (4 / 3 * intersection / union).item()
    
    def cal_jaccard_index(self) -> float:
        '''
        Return Jaccard Index between two images
        '''
        jaccard = JaccardIndex(task="multiclass", num_classes=3)
        return jaccard(self.pred_argmax, self.truth).item()
    
    def cal_precision(self):
        precision = Precision(task="multiclass", average='macro', num_classes=3)
        return precision(self.pred_argmax, self.truth).item()
