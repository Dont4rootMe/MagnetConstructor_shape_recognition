import numpy as np
import torch
import os


model = torch.nn.Sequential(
    torch.nn.Linear(39, 60),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(60),
    torch.nn.Linear(60, 60),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(60),
    torch.nn.Linear(60, 60),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(60),
    torch.nn.Linear(60, 60),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(60),
    torch.nn.Linear(60, 2)
)

for layer in model.modules():
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight)

DATA = []
LABELS = []

negs = 0
pos = 0

for file in os.listdir('./data'):
    if file.startswith('negative'):
        fill_value = 0
    else:
        fill_value = 1
    dt = np.load(os.path.join('./data', file), allow_pickle=True)
    DATA.append(dt)
    LABELS += [fill_value] * len(dt)

    if fill_value == 0:
        negs += len(dt)
    else:
        pos += len(dt)

print(negs, pos, negs / pos)

TRAIN = torch.tensor(np.concatenate(DATA, axis=0), dtype=torch.float32)
LABELS = torch.tensor(LABELS, dtype=torch.float32)
LABELS = torch.stack([torch.abs(LABELS - 1), LABELS], dim=-1)

# print(torch.unique(LABELS, return_counts=True))


optim = torch.optim.Adamax(model.parameters())
loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([3., 1.]))

for i in range(500):
    optim.zero_grad()

    logits = model(TRAIN)

    loss = loss_func(logits, LABELS)
    loss.backward()

    optim.step()
    if i % 10 == 0:
        print(f'step {i}: loss={loss.item()}')

torch.save(model.state_dict(), 'segmentation_model.pt')