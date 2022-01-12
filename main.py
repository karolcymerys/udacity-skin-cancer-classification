import math
import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision import models

LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 24
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
train_dataset = ImageFolder('./train', transform=transform)
valid_dataset = ImageFolder('./valid', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

model = models.googlenet(pretrained=True)
model.fc = nn.Linear(1024, 3)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.001)
criteria = nn.CrossEntropyLoss()

loss_history = {
    'train_loss': [],
    'validation_loss': [],
    'validation_accuracy': []
}
BATCHES = math.ceil(len(train_dataset) / BATCH_SIZE)
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_id, (features, labels) in enumerate(train_loader):
        batch_id += 1
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(features)
        loss = criteria(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item()) * int(features.shape[0])
        sys.stdout.write(f"\rProgress: {batch_id}/{BATCHES}\tTraining loss: {train_loss / (batch_id * BATCH_SIZE)}")

    model.eval()
    val_loss = 0.0
    accuracy = 0.0
    for features, labels in valid_loader:
        features, labels = features.to(device), labels.to(device)
        output = model.forward(features)
        loss = criteria(output, labels)
        val_loss += float(loss.item()) * int(features.shape[0])
        accuracy += int(torch.sum(torch.max(output, dim=1)[1] == labels))
    sys.stdout.write(f'\rEpoch: {epoch + 1}\t'
                     f'Train loss: {train_loss / len(train_dataset)}\t'
                     f'Validation loss: {val_loss / len(valid_dataset)}\t'
                     f'Validation acc: {accuracy / len(valid_dataset)*100}%\n')
    loss_history['train_loss'].append(train_loss / len(train_dataset))
    loss_history['validation_loss'].append(val_loss / len(valid_dataset))
    loss_history['validation_accuracy'].append(accuracy / len(valid_dataset))

    if len(loss_history['validation_loss']) > 2 and \
            torch.mean(torch.tensor(loss_history['validation_loss'][-3:-1])).item() < loss_history['validation_loss'][-1] and \
            torch.mean(torch.tensor(loss_history['validation_accuracy'][-3:-1])).item() < loss_history['validation_accuracy'][-1]:
        print('Training data overfitting detected. Finishing.')
        torch.save(model.state_dict(), 'densenet121.pth')
        break

plt.plot(loss_history['train_loss'], label='Training loss')
plt.plot(loss_history['validation_loss'], label='Validation loss')
plt.legend(frameon=False)
plt.show()
