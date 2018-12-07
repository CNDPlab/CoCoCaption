import torch as t
import ipdb


def transformeri_celoss(output, target):
    device = output.device
    vocab_size = output.size(2)
    batch_size = target.size(0)
    loss = t.nn.functional.cross_entropy(
        output.view(-1, vocab_size),
        t.cat([target, t.zeros((batch_size, 1), dtype=t.long, device=device)], -1).view(-1),
        ignore_index=0
    )
    return loss


def transformer_celoss(output, target):
    device = output.device
    vocab_size = output.size(2)
    batch_size = target.size(0)
    loss = t.nn.functional.cross_entropy(
        output.view(-1, vocab_size),
        t.cat([target, t.zeros((batch_size, 1), dtype=t.long, device=device)], -1)[:, 1:].contiguous().view(-1),
        ignore_index=0
    )
    return loss

