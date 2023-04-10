import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def loss_function(y, t, drop_rate):
    loss_fn = nn.CrossEntropyLoss()
    label_onehot = np.eye(4, dtype=np.uint8)[t]
    # loss = loss_fn(y, t, reduce = False)
    label_onehot = torch.from_numpy(label_onehot).type(torch.FloatTensor)
    loss = F.binary_cross_entropy_with_logits(y, label_onehot, reduce=False)

    loss_mul = loss * label_onehot
    ind_sorted = np.argsort(loss_mul.cpu().data)
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], label_onehot[ind_update])

    return loss_update


def loss_function_a(y, t, alpha):
    label_onehot = np.eye(4, dtype=np.uint8)[t]
    label_onehot = torch.from_numpy(label_onehot).type(torch.FloatTensor)
    loss = F.binary_cross_entropy_with_logits(y, label_onehot, reduce=False)
    y_ = torch.sigmoid(y).detach()
    weight = torch.pow(y_, alpha) * label_onehot + torch.pow((1 - y_), alpha) * (1 - label_onehot)
    loss_ = loss * weight
    loss_ = torch.mean(loss_)
    return loss_
