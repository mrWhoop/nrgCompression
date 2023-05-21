import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from AutoencoderThird import Autoencoder


arr = np.load('skull3d.npy')

sub_shape = (16, 16, 16)

subarrays = {}
for x in range(0, arr.shape[0], sub_shape[0]):
    for y in range(0, arr.shape[1], sub_shape[1]):
        for z in range(0, arr.shape[2], sub_shape[2]):
            sub_arr = arr[x:x+sub_shape[0], y:y+sub_shape[1], z:z+sub_shape[2]]
            sub_tensor = torch.from_numpy(sub_arr).float()
            subarrays[(x, y, z)] = sub_tensor.unsqueeze(0)


# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    for tens in subarrays.keys():
        optimizer.zero_grad()
        outputs = autoencoder(subarrays[tens])
        loss = criterion(outputs, subarrays[tens])
        loss.backward()
        optimizer.step()

torch.save(autoencoder.state_dict(), 'autoencoderThirdLCR54skull.pth')

print('Done.')
