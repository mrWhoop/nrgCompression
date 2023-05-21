import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from AutoencoderThird import Autoencoder
import sys

np.set_printoptions(threshold=sys.maxsize)

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoderThirdLCR54skull.pth'))

arr = np.load('skull3d.npy')

sub_shape = (16, 16, 16)

subarrays = {}
for x in range(0, arr.shape[0], sub_shape[0]):
    for y in range(0, arr.shape[1], sub_shape[1]):
        for z in range(0, arr.shape[2], sub_shape[2]):
            sub_arr = arr[x:x+sub_shape[0], y:y+sub_shape[1], z:z+sub_shape[2]]
            sub_arr = sub_arr / 256
            # sub_tensor = torch.from_numpy(sub_arr).float()
            sub_tensor = torch.from_numpy(sub_arr).float()
            subarrays[(x, y, z)] = sub_tensor.unsqueeze(0)

runningLoss = 0

compressed = {}
reconstructed = {}

for tensor in subarrays.keys():
    # Pass the input tensor through the autoencoder to obtain the compressed representation
    compressed_tensor = autoencoder.encoder(subarrays[tensor])
    compressed[tensor] = compressed_tensor
    # Use the decoder to reconstruct the original data from the compressed representation
    reconstructed_tensor = autoencoder.decoder(compressed_tensor)
    reconstructed[tensor] = reconstructed_tensor
    # Calculate the reconstruction loss between the original input and the reconstructed output
    loss = nn.MSELoss()(subarrays[tensor], reconstructed_tensor)
    runningLoss += loss.item()
    # print(tensor, ': ', loss.item())

print('running loss: ', runningLoss / len(subarrays.keys()))

print(sys.getsizeof(compressed))

# Create new array and fill it with the subarrays
new_arr = np.zeros_like(arr, dtype=float)
for pos, sub_arr in reconstructed.items():
    x, y, z = pos
    new_arr[x:x+sub_shape[0], y:y+sub_shape[1], z:z+sub_shape[2]] = sub_arr.squeeze(0).detach().numpy()


new_arr = new_arr * 255
new_arr = new_arr.astype(int)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.nonzero(new_arr)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot')

ax.scatter(x, y, z, s=0.1, c=new_arr[x, y, z], cmap='jet', vmin=0, vmax=255)

plt.show()
