from pycrcnn.convolutional.convolutional_layer import Conv2d, Conv1d
from pycrcnn.functional.average_pool import AveragePoolLayer
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.square_layer import SquareLayer
from pycrcnn.linear.linear_layer import LinearLayer


class Sequential:
    """
    Class which mimics PyTorch Sequential models.
    This class will be used as a container for the
    PyCrCNN layers.
    """
    layers = []

    """
    Given a PyTorch sequential model, create the corresponding PyCrCNN model.
    """
    def __init__(self, HE, model):
        self.HE = HE

        # Define builders for every possible layer

        def conv2d(layer):
            if layer.bias is None:
                bias = None
            else:
                bias = layer.bias.detach().numpy()

            return Conv2d(HE, weights=layer.weight.detach().numpy(),
                          stride=layer.stride,
                          padding=layer.padding,
                          bias=bias)

        def conv1d(layer):
            if layer.bias is None:
                bias = None
            else:
                bias = layer.bias.detach().numpy()

            return Conv1d(HE, weights=layer.weight.detach().numpy(),
                          stride=layer.stride,
                          padding=layer.padding,
                          bias=bias)

        def lin_layer(layer):
            if layer.bias is None:
                bias = None
            else:
                bias = layer.bias.detach().numpy()
            return LinearLayer(HE, layer.weight.detach().numpy(),
                               bias)

        def avg_pool_layer(layer):
            # This proxy is required because in PyTorch an AvgPool2d can have kernel_size, stride and padding either of
            # type (int, int) or int, unlike in Conv2d
            kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size,
                                                                               int) else layer.kernel_size
            stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride
            padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

            return AveragePoolLayer(HE, kernel_size, stride, padding)

        def flatten_layer(layer):
            return FlattenLayer()

        def square_layer(layer):
            return SquareLayer(HE)

        # Maps every PyTorch layer type to the correct builder
        options = {"Conv2d": conv2d,
                   "Conv1d": conv1d,
                   "Linear": lin_layer,
                   "Flatte": flatten_layer,
                   "AvgPoo": avg_pool_layer,
                   "Square": square_layer
                   }

        self.layers = [options[str(layer)[0:6]](layer) for layer in model]

    def __call__(self, x, debug=False):
        for layer in self.layers:
            x = layer(x)
            if debug:
                print(f"Passed layer: {layer}")
                print(f"Noise Budget: {self.HE.noise_budget(x.item(0))}")
        return x

