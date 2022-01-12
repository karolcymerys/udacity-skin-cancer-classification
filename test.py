import csv
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

OUTPUT_FILE = 'output.csv'
FILENAME = './densenet121.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 3)
model = model.to(device)
model.load_state_dict(torch.load(FILENAME))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_dataset = ImageFolder('./test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with open(OUTPUT_FILE, 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Id', 'task_1', 'task_2'])

    for batch_id, (features, _) in enumerate(test_loader):
        sys.stdout.write(f"\rEvaluation Progress: {batch_id}/{len(test_dataset)}")
        features = features.to(device)
        img_name = test_dataset.imgs[batch_id]
        output = model.forward(features)
        exp_y = torch.exp(output[0])
        y = exp_y/torch.sum(exp_y)
        csv_writer.writerow([img_name[0], y[test_dataset.class_to_idx['melanoma']].item(), y[test_dataset.class_to_idx['seborrheic_keratosis']].item()])