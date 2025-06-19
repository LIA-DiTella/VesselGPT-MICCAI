#!/usr/bin/env python
import torch.nn.functional as F
import torch.nn as nn

'''
def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """
    rec_loss = nn.L1Loss()(pred, target)
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    quant_loss = quant_loss.mean()
    return quant_loss * quant_loss_weight + rec_loss, [rec_loss, quant_loss]'''

def calc_vq_loss(pred, target, quant_loss, quant_loss_weight, warmup_epochs=10, current_epoch=0):
    rec_loss = nn.L1Loss()(pred, target)
    quant_loss_weight = quant_loss_weight if current_epoch >= warmup_epochs else quant_loss_weight
    total_loss = rec_loss + quant_loss_weight * quant_loss
    #print("rec_loss", rec_loss)
    #print("quant_loss", quant_loss)
    #print("weight", quant_loss_weight)
    #print("total_loss", total_loss)
    return total_loss, (rec_loss, quant_loss)

def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss
