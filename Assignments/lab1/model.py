import torch
import torch.nn as nn
import torch.nn.functional as F


class autoencoderMLP4Layer(nn.Module):

    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = N_input // 2
        # initialize autoencoder layer
        self.encoder_layer1 = nn.Linear(N_input, N2)
        self.encoder_layer2 = nn.Linear(N2, N_bottleneck)
        self.decoder_layer1 = nn.Linear(N_bottleneck, N2)
        self.decoder_layer2 = nn.Linear(N2, N_output)

    def encode(self, X):
        x = F.relu(self.encoder_layer1(X))
        x = F.relu(self.encoder_layer2(x))
        return x

    def decode(self, X):
        x = F.relu(self.decoder_layer1(X))
        x = torch.sigmoid(self.decoder_layer2(x))
        return x

    def forward(self, X):
        return self.decode(self.encode(X))
