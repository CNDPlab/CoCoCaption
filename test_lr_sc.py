import torch as t
import matplotlib.pyplot as plt



model = t.nn.Embedding(3,4)
optim = t.optim.Adam(model.parameters())




sc = t.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', 0.7, verbose=True,patience=0,)
score = [0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.7]


for i in score:
    print(i)
    sc.step(i)