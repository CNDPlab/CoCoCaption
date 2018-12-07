import torch as t
import ipdb


def celoss(target, output):
    criterion = t.nn.CrossEntropyLoss(ignore_index=0)
    loss = criterion(output.transpose(1,2), target[:,0:(output.shape[1])].long().cuda())
    return loss