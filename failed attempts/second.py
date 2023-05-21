import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16x16x16
            nn.Conv3d(32, 16, kernel_size=3, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8x8x8
            nn.Conv3d(16, 8, kernel_size=3, padding=1),  # 8x8x8
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 4x4x4
            nn.Conv3d(8, 4, kernel_size=3, padding=1),  # 8x8x8
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x8
            nn.ReLU(),
            nn.ConvTranspose3d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8x8
            nn.ReLU(),
            nn.ConvTranspose3d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x16
            nn.ReLU(),
            nn.ConvTranspose3d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x32
            nn.ReLU(),
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x16
            nn.ReLU(),
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


arr = np.load('../foot3d.npy')

#
# input_tensor = torch.from_numpy(input_array).float()
#
# input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
#
# dataset = MyDataset(input_tensor)

# Create a data loader for the dataset
# batch_size = 8
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Define subarray shape
sub_shape = (8, 8, 8)

subarrays = {}
for x in range(0, arr.shape[0], sub_shape[0]):
    for y in range(0, arr.shape[1], sub_shape[1]):
        for z in range(0, arr.shape[2], sub_shape[2]):
            sub_arr = arr[x:x+sub_shape[0], y:y+sub_shape[1], z:z+sub_shape[2]]
            sub_tensor = torch.from_numpy(sub_arr).float()
            subarrays[(x, y, z)] = sub_tensor

tensor = torch.cat([subarrays[pos].unsqueeze(0) for pos in subarrays], dim=0)

print(tensor)

dataset = MyDataset(tensor)

# Create a data loader for the dataset
# batch_size = 32
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# Train the autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    #for batch_idx, (inputs, targets) in enumerate(data_loader):
    for tens in tensor:
        optimizer.zero_grad()
        outputs = autoencoder(tens)
        loss = criterion(outputs, tens)
        loss.backward()
        optimizer.step()
    #    running_loss += loss.item()
    # epoch_loss = running_loss / (len(dataset) / batch_size)
    # print('Epoch {} Loss: {:.6f}'.format(epoch+1, epoch_loss))


# Pass the input tensor through the autoencoder to obtain the compressed representation
compressed_tensor = autoencoder.encoder(tensor)

# Use the decoder to reconstruct the original data from the compressed representation
reconstructed_tensor = autoencoder.decoder(compressed_tensor)

# Calculate the reconstruction loss between the original input and the reconstructed output
loss = nn.MSELoss()(tensor, reconstructed_tensor)

# Print the shape and loss of the reconstructed tensor
print(reconstructed_tensor.shape, loss.item())

torch.save(autoencoder.state_dict(), "second.pth")

print(tensor)

print(compressed_tensor)

print(reconstructed_tensor)


