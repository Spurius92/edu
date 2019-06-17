import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU,
)
criterion = nn.CrossEntropyLoss
optimizer = optim.RMSprop(model.parameters)


def train(X, y, model, criterion, optimizer, epochs=1, device=None):
    if device is not None:
        model.to(device)

    model._zero_grad()
    losses = {}
    for i in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        model._zero_grad()
        losses[i] = loss
    return losses
