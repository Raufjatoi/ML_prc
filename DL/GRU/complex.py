import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create a synthetic dataset for sequence prediction
def create_sequence_data(seq_length, num_sequences):
    X = []
    Y = []
    for _ in range(num_sequences):
        seq = np.sin(np.linspace(0, 3 * np.pi, seq_length + 1))
        X.append(seq[:-1])  # Input sequence
        Y.append(seq[1:])   # Target sequence (next step)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Define the GRU model
class SequenceGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SequenceGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # Initial hidden state
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 1    # Number of features in input
hidden_size = 32  # Number of hidden units in GRU
num_layers = 2    # Number of GRU layers
output_size = 1   # Output size (one feature per time step)
seq_length = 20   # Length of each input sequence
num_sequences = 100  # Number of sequences in the dataset
num_epochs = 200  # Training epochs
learning_rate = 0.01

# Create dataset
X, Y = create_sequence_data(seq_length, num_sequences)
X = X.unsqueeze(-1)  # Reshape to (batch_size, seq_length, input_size)

# Create the model
model = SequenceGRU(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):e
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y.unsqueeze(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")
