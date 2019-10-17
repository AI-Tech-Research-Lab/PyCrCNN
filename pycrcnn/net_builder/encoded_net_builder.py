from pycrcnn.convolutional.convolutional_layer import ConvolutionalLayer
from pycrcnn.functional.average_pool import AveragePoolLayer
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.linear.linear_layer import LinearLayer


def build_from_pytorch(HE, net, rencrypt_positions):
    """Given a PyTorch sequential net in a .pt/.pth file, returns
    an ordered list of encoded layers on which is possible
    to apply an encrypted computation.

    Parameters
    ----------
    HE : Pyfhel
        Pyfhel object
    net: nn.Sequential
        PyTorch model in nn.Sequential form
    rencrypt_position: int
        Number of the layer after which a rencrypt_layer
        will be inserted

    Returns
    -------
    encoded_layers: list
        Ordered list of encoded layers which reflects
        the PyTorch model given in input
    """

    # Define builders for every possible layer
    def conv_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()

        return ConvolutionalLayer(HE, layer.weight.detach().numpy(),
                                  layer.stride[0],
                                  layer.stride[1],
                                  bias)

    def lin_layer(layer):
        if layer.bias is None:
            bias = None
        else:
            bias = layer.bias.detach().numpy()
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           bias)

    def avg_pool_layer(layer):
        return AveragePoolLayer(HE, layer.kernel_size, layer.stride)

    def flatten_layer(layer):
        return FlattenLayer()

    # Maps every PyTorch layer type to the correct builder
    options = {"Conv": conv_layer,
               "Line": lin_layer,
               "Flat": flatten_layer,
               "AvgP": avg_pool_layer
               }

    encoded_layers = [options[str(layer)[0:4]](layer) for layer in net]
    [encoded_layers.insert(i, RencryptionLayer(HE)) for i in rencrypt_positions]
    return encoded_layers
