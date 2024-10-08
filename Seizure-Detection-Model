import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 22), stride=(1, 1), padding="same")
        self.norm1 = nn.LayerNorm(normalized_shape=(256,))
        self.relu1 = nn.ReLU()
        
        # Additional convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(1, 1), padding="same")
        self.norm2 = nn.LayerNorm(normalized_shape=(256,))
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(10, 1), stride=(1, 1), padding="same")
        self.norm3 = nn.LayerNorm(normalized_shape=(128,))
        self.relu3 = nn.ReLU()
        
        # Third convolutional block
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(20, 1), stride=(1, 1), padding="same")
        self.norm4 = nn.LayerNorm(normalized_shape=(128,))
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 2))
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(11264, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = x.permute(0, 1, 3, 2)
        x = self.norm1(x)
        x = x.permute(0, 1, 3, 2)
        x = self.relu1(x)
        
        # Additional convolutional block
        x = self.conv2(x)
        x = x.permute(0, 1, 3, 2)
        x = self.norm2(x)
        x = x.permute(0, 1, 3, 2)
        x = self.relu2(x)
        x = self.maxpool1(x)
        
        # Second convolutional block
        x = self.conv3(x)
        x = x.permute(0, 1, 3, 2)
        x = self.norm3(x)
        x = x.permute(0, 1, 3, 2)
        x = self.relu3(x)
        
        # Third convolutional block
        x = self.conv4(x)
        x = x.permute(0, 1, 3, 2)
        x = self.norm4(x)
        x = x.permute(0, 1, 3, 2)
        x = self.relu4(x)
        x = self.maxpool2(x)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

# model in cpu
model = cnn().to("cuda:0")
