import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x, x


# create a sample input tensor
input_array = np.load('../engine3d.npy')

input_tensor = torch.from_numpy(input_array).float()

input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

dataset = MyDataset(input_tensor)

# Create a data loader for the dataset
batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / (len(dataset) / batch_size)
    print('Epoch {} Loss: {:.6f}'.format(epoch+1, epoch_loss))

# Pass the input tensor through the autoencoder to obtain the compressed representation
compressed_tensor = autoencoder.encoder(input_tensor)

# Use the decoder to reconstruct the original data from the compressed representation
reconstructed_tensor = autoencoder.decoder(compressed_tensor)

# Calculate the reconstruction loss between the original input and the reconstructed output
loss = nn.MSELoss()(input_tensor, reconstructed_tensor)

# Print the shape and loss of the reconstructed tensor
print(reconstructed_tensor.shape, loss.item())
