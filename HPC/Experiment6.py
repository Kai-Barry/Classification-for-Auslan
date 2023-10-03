import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import time


Locations = {}

f = open("keyBindsLocation.txt", "r")
line = f.readline()
while (line):
    try:
        if line.split("_|_")[1] == 'None\n':
            Locations[line.split("_|_")[0]] = None
        else:
            Locations[line.split("_|_")[0]] = line.split("_|_")[1]
    except:
        i = 1
    line = f.readline()
f.close()

print(len(Locations.items()))
print(len([i for i in Locations.values() if not i == None]))

usability = {}

f = open("useability.txt", "r")
line = f.readline()
while (line):
    try:
        if line.split("_|_")[1] == 'None\n':
            usability[line.split("_|_")[0]] = None
        else:
            usability[line.split("_|_")[0]] = int(line.split("_|_")[1])
    except:
        if line != '\n':
            print(line)
    line = f.readline()
f.close()

label_keys = {"Head\n":0, "Eye\n":1, "Forehead\n":2, "Nose\n":3, 
              "Ear\n":4, "Mouth\n":5, "Chin\n":6, 
              "Neck\n":7, "Shoulders\n":8, 
              "ArmPit\n":9, "Chest\n":10, "Waist\n": 11,
              "Back\n":12, "Thigh\n":13, "Stomach\n": 14,
              "Arm\n":15, "Wrist\n":16, "Hand\n": 17,
              "Neutral\n": 18}

X = []
Y = []
imgDataLocation = "./Data/"
dirData = os.listdir(imgDataLocation)
title = []

start = time.time()
for i, data in enumerate(dirData):
    if data.split('_')[0] in Locations and Locations[data.split('_')[0]] is not None and usability[data[:-3] +'.mp4'] >= 3:
        fileLocation = imgDataLocation + data + '/'
        if len(title) == 0 or title[-1] != data[:-3] :
            title.append(data[:-3])
            dirImg = os.listdir(fileLocation)
            images = []
            for img in dirImg:
                image = Image.open(fileLocation + img)
                image = image.convert('L')
                images.append(np.array(image))
            images = np.array(images)
            X.append([images])
            Y.append(label_keys[Locations[data.split('_')[0]]])
        else :
            dirImg = os.listdir(fileLocation)
            images = []
            for img in dirImg:
                image = Image.open(fileLocation + img)
                image = image.convert('L')
                images.append(np.array(image))
                # print(np.array(image).shape)
            images = np.array(images)
            # print(images.shape)
            X[-1].append(images)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18  # ResNet-18 architecture

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
batch_size = 32




# Define the dimensions based on your data
input_dim = 3 * 40 * 128 * 128  # 40 frames, 33 points, 3 dimensions
hidden_dim = 256
output_dim = len(label_keys)

class BasicNeuralModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicNeuralModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(input_dim*4, int(input_dim/480))
        # self.fc2 = nn.Linear(int(input_dim/480), hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Instantiate the model
model = BasicNeuralModel(input_dim, hidden_dim, output_dim)
model = model.to(device)

num_epochs =  600
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Set the model in training mode
model.train()

for epoch in range(num_epochs):    
    for batch_inputs, batch_targets in train_loader:  # You need to create a DataLoader for your data
        batch_inputs = batch_inputs.float().to(device)
        batch_targets  = batch_targets.to(device)
        batch_targets = batch_targets.view(-1, 1) 
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_inputs)
        
        # Compute the loss
        loss = criterion(outputs, batch_targets)
        
        # Backpropagation
        loss.backward()
        
        # Update the weights
        optimizer.step()
    
    # Print the loss for this epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'BasicNeuralModelExperiment5.pth')

def calculate_accuracy(predictions, labels):
    # Calculate overall accuracy
    total_samples = labels.size(0)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / total_samples

    # Calculate accuracy for each label
    label_accuracy = {}
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        label_indices = (labels == label).nonzero().squeeze()
        label_predictions = predictions[label_indices]
        correct_label_predictions = (label_predictions == label).sum().item()
        total_label_samples = len(label_indices)
        label_accuracy[label.item()] = correct_label_predictions / total_label_samples

    return accuracy, label_accuracy

# Set the model to evaluation mode
model.eval()

# Forward pass to get predictions
with torch.no_grad():
    test_predictions = model(X_test_tensor)

# Convert predictions to class labels (assuming your output is regression-like)
test_predictions_labels = torch.argmax(test_predictions, dim=1)

# Calculate accuracy
overall_accuracy, label_accuracy = calculate_accuracy(test_predictions_labels, y_test_tensor)

print(f'Overall Accuracy: {overall_accuracy:.4f}')
print('Label Accuracy:')
for label, acc in label_accuracy.items():
    print(f'Label {label}: {acc:.4f}')


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
        
#         # First convolution layer
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels) 
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Shortcut connection if dimensions change
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
        
#     def forward(self, x):
#         # Main path
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         # Shortcut path
#         shortcut = self.shortcut(x)
        
#         # Residual connection
#         out += shortcut
#         out = self.relu(out)
        
#         return out
    

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
        
#         # Initial convolutional layer
#         self.conv1 = nn.Conv2d(40, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
        
#         # Create layers with residual blocks
#         self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        
#         # Global average pooling and fully connected layer
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, out_channels, num_blocks, stride):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for _ in range(1, num_blocks):
#             layers.append(block(out_channels, out_channels, stride=1))
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
        
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return out

# num_epochs =  100

# learning_rate = 0.1

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# #Piecwise Linear Schedule
# sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=learning_rate, step_size_down=15, mode='triangular', verbose=False)
# sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/learning_rate, end_factor=0.005/5, verbose=False)
# scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

# n_total_steps = len(train_loader)

# # Set the model in training mode
# model.train()

# for epoch in range(num_epochs):    
#     for batch_inputs, batch_targets in train_loader:  # You need to create a DataLoader for your data
#         batch_inputs = batch_inputs.float().to(device)
#         batch_targets  = batch_targets.to(device).to(torch.int64)

#         # Forward pass
#         outputs = model(batch_inputs) 
#         loss = criterion(outputs, batch_targets) 

        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     # Print the loss for this epoch
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')