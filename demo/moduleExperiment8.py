import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels) 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
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
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Create layers with residual blocks
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*5, num_classes)

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
    
def get_resNet_model(output_dim):
    num_classes = output_dim
    model = ResNet(ResidualBlock, [2,2,2,2], num_classes).to(device)
    model.to(device)
    return model
    
def load_model(model, location):
    model.load_state_dict(torch.load(location))

def convert_tensor(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64).to(device)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def reshape_x(X_train, X_test):
    return X_train.reshape(67, 1, 40, 128, 128), X_test.reshape(33, 1, 40, 128, 128)


def get_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=1):
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def model_accuracy(testloader, model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).int()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

def get_connections_list():
    return np.loadtxt('./experiment8/connections_list.npy')

def get_example(location='experiment8/unprocessed/MONKEY_AAPB2c6iii_20190_21240.npy', 
                shapeLocation='experiment8/unprocessed/MONKEY_AAPB2c6iii_20190_21240_shape.npy'):
    coorLoad = np.loadtxt(location)
    coorShape= np.loadtxt(shapeLocation)
    return coorLoad.reshape(coorLoad.shape[0], coorLoad.shape[1] // int(coorShape[2]), int(coorShape[2]))


def convert_one_video(video, size):
    connections_list = get_connections_list()
    newVideo = []
    for frame in video:
        newVideo.append(frame[0:25])
    video = np.array(newVideo)

    padding = 0
    x_max = max(np.array(video).T[0].reshape(1,-1)[0]) + padding/2
    x_min = min(np.array(video).T[0].reshape(1,-1)[0]) - padding/2
    x_range = abs(x_max - x_min)
    y_max = max(np.array(video).T[1].reshape(1,-1)[0]) + padding/2
    y_min = min(np.array(video).T[1].reshape(1,-1)[0]) - padding/2
    y_range = abs(y_max - y_min)
    z_max = max(np.array(video).T[2].reshape(1,-1)[0]) + padding/2
    z_min = min(np.array(video).T[2].reshape(1,-1)[0]) - padding/2
    z_range = abs(z_max - z_min)

    plotFrame = False

    xy_frames = []
    xz_frames = []
    yz_frames = []

    for frame in video:
        frame  = frame.T
        x = frame[0]
        y = frame[1]
        z = frame[2]
        xy = np.zeros((size,size))
        xz = np.zeros((size,size))
        yz = np.zeros((size,size))

        coord2d = [(xy, (x_max, x_min, x_range, x), (y_max, y_min, y_range, y)), (xz,(x_max, x_min, x_range, x), (z_max, z_min, z_range, z)),(yz, (z_max,z_min, z_range, z), (y_max,y_min, y_range, y))]
        for axis2d, axis1, axis2 in coord2d:
            axis1_max, axis1_min, axis1_range, data1 = axis1
            axis2_max, axis2_min, axis2_range,data2 = axis2
            max_edge, big_axis = max((axis1_range, 'a1'), (axis2_range, 'a2'))
            if axis1_min < 0:
                axis1_offset = abs(axis1_min)
            else:
                axis1_offset = -1 * abs(axis1_min)
            if axis2_min < 0:
                axis2_offset = abs(axis2_min)
            else:
                axis2_offset = -1 * abs(axis2_min)

            multiplier = (size - 1)/max_edge
            for connection in connections_list:
                p1, p2 = connection
                p1, p2 = int(p1), int(p2)
                point_coor1 = ((data1[p1] + axis1_offset) * multiplier, (data2[p1] + axis2_offset) * multiplier)
                point_coor2 = ((data1[p2] + axis1_offset) * multiplier, (data2[p2] + axis2_offset) * multiplier)
                
                m = (point_coor2[1]- point_coor1[1])/(point_coor2[0]- point_coor1[0])
                lineX = lambda x : m * x - m * point_coor1[0] + point_coor1[1]
                lineY = lambda y : ((y - point_coor1[1])/m) + point_coor1[0]

                # Plot
                if plotFrame:
                    plt.plot([point_coor1[0], point_coor2[0]],[point_coor1[1], point_coor2[1]])

                # Point Location
                a1p1Index = int(point_coor1[0])
                a2p1Index = int(point_coor1[1])
                a1p2Index = int(point_coor2[0])
                a2p2Index = int(point_coor2[1])
                
                # create line between y
                if a2p1Index < a2p2Index:
                    for y_index in range(a2p1Index,a2p2Index + 1):
                        if y_index >= point_coor1[1] and y_index <= point_coor2[1]:
                            x_index = int(lineY(y_index))
                            axis2d[y_index, x_index] = 1
                            if plotFrame:
                                plt.scatter(x_index, y_index, c="red")
                else:
                    for y_index in range(a2p2Index,a2p1Index + 1):
                        if y_index >= point_coor2[1] and y_index <= point_coor1[1]:
                            x_index = int(lineY(y_index))
                            axis2d[y_index, x_index] = 1
                            if plotFrame:
                                plt.scatter(x_index, y_index, c="red")

                if a1p1Index < a1p2Index:
                    for x_index in range(a1p1Index,a1p2Index + 1):
                        if x_index >= point_coor1[0] and x_index <= point_coor2[0]:
                            y_index = int(lineX(x_index))
                            axis2d[y_index, x_index] = 1
                            if plotFrame:
                                plt.scatter(x_index, y_index, c="blue")
                else:
                    for x_index in range(a1p2Index, a1p1Index + 1):
                        if x_index >= point_coor2[0] and x_index <= point_coor1[0]:
                            y_index = int(lineX(x_index))
                            axis2d[y_index, x_index] = 1
                            if plotFrame:
                                plt.scatter(x_index, y_index, c="blue")
        xy_frames.append(xy)
        xz_frames.append(xz)
        yz_frames.append(yz)
    xy_frames = np.array(xy_frames)
    xz_frames = np.array(xz_frames)
    yz_frames = np.array(yz_frames)
    return xy_frames, xz_frames, yz_frames