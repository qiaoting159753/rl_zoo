from agents.tqc.HyperMLP import HyperMLP
from agents.utils import MLP
import torch.nn.functional as F
import numpy as np
import torch

net = HyperMLP(input_size=2, output_size=1)
# net = MLP(2,[64, 64], 1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def function(x_x):
    return x_x[0] * 5 + x_x[1]


def create_batch(batch_size=100):
    x_s = []
    y_s = []
    for _ in range(batch_size):
        x = np.random.rand(2, )
        x_s.append(x)
        y = function(x)
        y_s.append(y)
    x_s = torch.FloatTensor(np.array(x_s))
    y_s = torch.FloatTensor(np.array(y_s)).unsqueeze(1)
    return x_s, y_s


for i in range(1000):
    # Training
    x_s, y_s = create_batch()
    pred = net(x_s)
    loss = F.mse_loss(pred, y_s)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluating.
    x_s, y_s = create_batch()
    pred = net(x_s)
    loss = F.mse_loss(pred, y_s)
    print(loss.item())
