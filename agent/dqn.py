import torch
import torch.nn as nn


# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=2),  # First conv layer
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  # Second conv layer
            nn.ReLU(),
        )

        # Adjust for 32x32 input size
        convw = self.conv2d_size_out(self.conv2d_size_out(16, 5, 2), 5, 2)
        convh = convw  # Assuming square input image

        linear_input_size = convw * convh * 16
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_size, 64),  # First linear layer
            nn.ReLU(),
            nn.Linear(64, num_actions),  # Second linear layer outputs number of actions
        )

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_layers(x)
