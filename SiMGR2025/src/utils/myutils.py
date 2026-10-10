import torch
import torch.nn as nn
import math


class MYEmbLoss():
    def __init__(self):
        pass
    def calc_loss(self, user_emb, pos_item_emb, neg_item_emb):
        len_ = user_emb.shape[0]
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              pos_item_emb.norm(2).pow(2) +
                              neg_item_emb.norm(2).pow(2)) / float(len_)
        return reg_loss

class RegLoss():
    def __init__(self):
        pass
    def calc_loss(self, input_parameters):
        reg_loss=None
        for param in input_parameters:
            if reg_loss is None:
                reg_loss = 0.5 * torch.sum(param**2)
            else:
                reg_loss = reg_loss + 0.5 * param.norm(2)**2
        return reg_loss