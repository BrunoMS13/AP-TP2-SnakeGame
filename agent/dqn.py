import torch
import torch.nn as nn


# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.input_channels = input_channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=2),  # First conv layer
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),  # Second conv layer
            nn.ReLU(),
        )

        # Adjust for 32x32 input size
        convw = self.conv2d_size_out(self.conv2d_size_out(32, 5, 2), 5, 2)
        convh = convw  # Assuming square input image

        linear_input_size = convw * convh * 16
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_size, 128),  # Adjusted linear layer size
            nn.ReLU(),
            nn.Linear(128, num_actions),  # Second linear layer outputs number of actions
        )

    def conv2d_size_out(self, size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_layers(x)

    def preprocess_input(self, input_tensor):
        # input_tensor shape: (batch_size, height, width, channels, num_frames)
        print(input_tensor.shape)
        # if batch size exists then process else return it
        if len(input_tensor.shape) == 4:
            return input_tensor.permute(0, 3, 1, 2)
        batch_size, height, width, channels, num_frames = input_tensor.shape
        # Permute to (batch_size, num_frames, height, width, channels)
        input_tensor = input_tensor.permute(0, 4, 1, 2, 3)
        # Reshape to (batch_size, channels * num_frames, height, width)
        input_tensor = input_tensor.reshape(batch_size, self.input_channels, height, width)
        return input_tensor
