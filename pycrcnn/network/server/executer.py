import torch
import torch.nn as nn
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)


def perform_computation(HE, enc_images, net, layers):
    # Or whetever the plain net is
    if net == "MNIST":
        plain_net = torch.load("./mnist.pt")
        plain_net.eval()
    if net == "SimpleModel":
        plain_net = torch.load("./SimpleModel.pt")
        plain_net.eval()

    # Choose how many layers encode
    plain_net = plain_net[min(layers):max(layers)+1]
    encoded_net = build_from_pytorch(HE, plain_net)

    for layer in encoded_net:
        enc_images = layer(enc_images)

    return enc_images

