# src/models/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvQNetwork(nn.Module):
    """Convolutional Q-Network for trading based on chart images."""
    
    def __init__(self, input_channels, output_size, seed=42):
        """Initialize network parameters.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB)
            output_size (int): Number of output actions
            seed (int): Random seed
        """
        super(ConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Calculate the size of the output from the last conv layer
        # This will depend on your input image size
        self.fc_input_size = self._get_conv_output_size(input_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Q-values for each action
        """
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer (no activation as these are Q-values)
        return self.fc3(x)
    
    def _get_conv_output_size(self, input_channels):
        """Calculate the size of the flattened features after convolutional layers.
        
        Args:
            input_channels (int): Number of input channels
            
        Returns:
            int: Size of the flattened features
        """
        # Create a dummy input tensor
        input_tensor = torch.zeros(1, input_channels, 224, 224)
        
        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(input_tensor)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten and return size
        return x.view(1, -1).size(1)