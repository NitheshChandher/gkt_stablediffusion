import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,image_size):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(3, 32, 3, 2, 0)
        self.cnn2 = nn.Conv2d(32, 64, 3, 2, 0)
        self.cnn3 = nn.Conv2d(64, 128, 3, 2, 0)
        self.cnn4 = nn.Conv2d(128, 256, 3, 2, 0)
        self.cnn5 = nn.Conv2d(256, 512, 3, 2, 0)
        self.cnn6 = nn.Conv2d(512, 1024, 3, 2, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
        
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, input_batch):
        x = self.cnn1(input_batch)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.cnn5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.cnn6(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.flatten(x)
        x = self.fc(x)
        out = self.sigmoid(x).view(-1)
        return out

"""
Build your own network below
"""   

