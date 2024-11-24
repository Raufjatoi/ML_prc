import torch
import torch.nn as nn

# Define the GRU model
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Hyperparameters
input_size = 1   # Number of features in input
hidden_size = 16 # Number of hidden units in GRU
output_size = 1  # Output size

# Create the model
model = SimpleGRU(input_size, hidden_size, output_size)

# Create a random input tensor (batch_size, sequence_length, input_size)
x = torch.randn(8, 5, input_size)  # Example with batch_size=8 and sequence_length=5
output = model(x)

print("Output shape:", output.shape)  # Should be [8, 1] for batch_size=8 and output_size=1
