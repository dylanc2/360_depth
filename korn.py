import numpy as np
import os
import kornia
import torch

b = []
n = []
for i in range(30):
    n.append(torch.eye(3))
K = torch.stack(n,dim=0)


for filename in sorted(os.listdir("outputs")):
    d = torch.tensor(np.load('outputs/'+filename))
    x = d[None, :]
    b.append(x)

x = torch.stack(b,dim=0)
print(x.shape)
print(K.shape)
print(kornia.geometry.depth.depth_to_3d(x, K).shape)