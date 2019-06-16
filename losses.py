import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

losses = np.load("tlosses.npy", allow_pickle=True)
losses = [float(ls) for ls in losses]
indices = []
for idx, l in enumerate(losses):
    indices.append(idx)
print(indices)
print(losses)
plt.plot(indices, losses, label="Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.savefig("tl.pdf")

