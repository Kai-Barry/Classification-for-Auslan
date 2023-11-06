import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def conver_tensor(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64).to(device)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def reshape_x(X_train, X_test):
    return X_train.reshape(X_train.shape[0], 40, 33, 3), X_test.reshape(X_test.shape[0], 40, 33, 3)


def get_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=32):
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


class BasicNeuralModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicNeuralModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Reshape to (batch_size, 40 * 33 * 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut path
        shortcut = self.shortcut(x)
        
        # Residual connection
        out += shortcut
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(40, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Create layers with residual blocks
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

def get_basicNN_model(output_dim):
    input_dim = 40 * 33 * 3  # 40 frames, 33 points, 3 dimensions
    hidden_dim = 256
    model = BasicNeuralModel(input_dim, hidden_dim, output_dim)
    model.to(device)
    return model

def get_resNet_model(output_dim):
    num_classes = output_dim
    model = ResNet(ResidualBlock, [2,2,2,2], num_classes).to(device)
    model.to(device)
    return model
    
def load_model(model, location):
    model.load_state_dict(torch.load(location))

def print_accuracy(model, label_keys, test_loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():

        correct = 0
        total = 0

        # Initialize a dictionary to store class-wise accuracy
        class_correct = {i: 0 for i in range(len(label_keys))}
        class_total = {i: 0 for i in range(len(label_keys))}

        for batch in test_loader:
            test_X, test_y = batch
            test_X, test_y = test_X.to(device), test_y.to(device)

            # Forward pass
            test_outputs = model(test_X)
            _, predicted = torch.max(test_outputs, 1)
            # Compute overall accuracy
            correct += (predicted == test_y).sum().item()
            total += test_y.size(0)
            # Compute class-wise accuracy
            for i in range(len(label_keys)):
                class_total[i] += (test_y == i).sum().item()
                class_correct[i] += (predicted == i)[test_y == i].sum().item()

        overall_accuracy = correct / total
        print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

        # Print class-wise accuracy
        for i in range(len(label_keys)):
            if class_total[i] == 0:
                print(f"Class {i} Accuracy: Nan")
            else:
                class_accuracy = class_correct[i] / class_total[i]
                print(f"Class {i} Accuracy: {class_accuracy:.4f}")

def compare_model(model1, model2, test_loader, train_loader):
    model1_test_correct = 0
    model2_test_correct = 0
    test_total = 0

    for batch in test_loader:
        test_X, test_y = batch
        test_X, test_y = test_X.to(device), test_y.to(device)

        # model1 test
        test_outputs_1 = model1(test_X)
        _, predicted1 = torch.max(test_outputs_1, 1)
        # Compute overall accuracy
        model1_test_correct += (predicted1 == test_y).sum().item()

        # model2 test
        test_outputs_2 = model2(test_X)
        _, predicted2 = torch.max(test_outputs_2, 1)
        # Compute overall accuracy
        model2_test_correct += (predicted2 == test_y).sum().item()

        test_total += test_y.size(0)
    
    model1_train_correct = 0
    model2_train_correct = 0
    train_total = 0

    for batch in train_loader:
        test_X, test_y = batch
        test_X, test_y = test_X.to(device), test_y.to(device)

        # model1 test
        test_outputs_1 = model1(test_X)
        _, predicted1 = torch.max(test_outputs_1, 1)
        # Compute overall accuracy
        model1_train_correct += (predicted1 == test_y).sum().item()

        # model2 test
        test_outputs_2 = model2(test_X)
        _, predicted2 = torch.max(test_outputs_2, 1)
        # Compute overall accuracy
        model2_train_correct += (predicted2 == test_y).sum().item()

        train_total += test_y.size(0)
    
    model1_test_accuracy = model1_test_correct / test_total
    model2_test_accuracy = model2_test_correct / test_total
    model1_train_accuracy = model1_train_correct / train_total
    model2_train_accuracy = model2_train_correct / train_total
    modelNames = ['Basic Neural Net', 'ResNet18']
    combined = np.array([modelNames,[model1_train_accuracy, model2_train_accuracy], [model1_test_accuracy,model2_test_accuracy]])
    return pd.DataFrame(combined.T, columns=['Model', 'Training Accuracy', 'Test Accuracy'])
    
def predict_all(loader, model):
    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    for batch in loader:
        test_X, test_y = batch
        test_X, test_y = test_X.to(device), test_y.to(device)

        # model test
        test_outputs_1 = model(test_X)
        _, predict = torch.max(test_outputs_1, 1)
        y_pred = torch.concatenate((y_pred, predict))
        y_true = torch.concatenate((y_true, test_y))
    return y_pred, y_true
    
def plot_confusion_deep(y_pred, y_true, label_keys, model_name, num_labels):
    inverted_label_keys = dict(map(reversed, label_keys.items()))
    label = [inverted_label_keys[i] for i in range(num_labels)]
    set(label_keys.keys())
    numberLabel = np.arange(num_labels)

    if not set(numberLabel).issubset(set(y_true)):
        missing_class = list(set(numberLabel) - set(y_true))
        y_true = np.concatenate((y_true, missing_class))
        y_pred = np.concatenate((y_pred, missing_class))

    cm = metrics.confusion_matrix(y_pred, y_true)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(model_name)
    plt.show()