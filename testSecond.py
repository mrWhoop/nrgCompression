import torch
from second import Autoencoder
import numpy as np

model = Autoencoder()

model.load_state_dict(torch.load('failed attempts/second.pth'))

input_array = np.load('foot3d.npy')

input_tensor = torch.from_numpy(input_array).float()

input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

# Pass the input tensor through the autoencoder to obtain the compressed representation
compressed_tensor = model.encoder(input_tensor)

# Use the decoder to reconstruct the original data from the compressed representation
reconstructed_tensor = model.decoder(compressed_tensor)
