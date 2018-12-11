import torch as t
import matplotlib.pyplot as plt



model = t.nn.Embedding(3,4)
optim = t.optim.Adam(model.parameters())
sc = t.optim.lr_scheduler.CosineAnnealingLR(optim, 5, 5e-4)


lrs = []

for i in range(100):
    sc.step()
    lrs.append(optim.param_groups[0]['lr'])

