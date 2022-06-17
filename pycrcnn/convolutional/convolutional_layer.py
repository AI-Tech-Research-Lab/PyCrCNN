import numpy as np

from pycrcnn.functional.padding import apply_padding
from pycrcnn.he.HE import CKKSPyfhel


class Conv2d:
    """
    A class used to represent a convolutional layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    weights : np.array( dtype=PyPtxt )
        Weights of the layer, aka filters in form
        [n_filters, n_layers, y, x]
    stride : (int, int)
        Stride (y, x)
    padding : (int, int)
        Padding (y, x)
    bias : np.array( dtype=PyPtxt ), default=None
        Biases of the layer, 1-D array


    Methods
    -------
    __init__(self, HE, weights, x_stride, y_stride, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che convolution operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using weights, biases and strides of the layer.
    """

    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = HE.encode_matrix(weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = HE.encode_matrix(bias)

    def __call__(self, t):
        # t = apply_padding(t, self.padding)
        # if isinstance(self.HE, CKKSPyfhel):
        #     for w in np.ravel(self.weights):
        #         for i in range(0, t[0][0][0][0].mod_level):
        #             self.HE.he.mod_switch_to_next(w)

        result = np.array([[np.sum([convolute2d(image_layer, filter_layer, self.stride)
                                    for image_layer, filter_layer in zip(image, _filter)], axis=0)
                            for _filter in self.weights]
                           for image in t])

        if self.bias is not None:
            result = np.array([[layer + bias for layer, bias in zip(image, self.bias)] for image in result])

        return result


def convolute2d(image, filter_matrix, stride):
    """Execute a convolution operation given an 2D-image, a 2D-filter
    and related strides.


    Parameters
    ----------
    image : np.array( dtype=PyCtxt )
        Encrypted image to execute the convolution on, in the form
        [y, x]
    filter_matrix : np.array( dtype=PyPtxt )
        Encoded weights to use in the convolution, in the form
        [y, x]
    stride : (int, int)
        Stride

    Returns
    -------
    result : np.array( dtype=PtCtxt )
        Encrypted result of the convolution, in the form
        [y, x]
    """
    x_d = len(image[0])
    y_d = len(image)
    x_f = len(filter_matrix[0])
    y_f = len(filter_matrix)

    y_stride = stride[0]
    x_stride = stride[1]

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    def get_submatrix(matrix, x, y):
        index_row = y * y_stride
        index_column = x * x_stride
        return matrix[index_row: index_row + y_f, index_column: index_column + x_f]

    return np.array(
        [[np.sum(get_submatrix(image, x, y) * filter_matrix) for x in range(0, x_o)] for y in range(0, y_o)])


class Conv1d:
    def __init__(self, HE, weights, stride=(1, 1), padding=(0, 0), bias=None):
        self.HE = HE
        self.weights = HE.encode_matrix(weights)
        self.stride = stride
        self.padding = padding
        self.bias = bias
        if bias is not None:
            self.bias = HE.encode_matrix(bias)

    def __call__(self, t):
        # t = apply_padding(t, self.padding)
        result = np.array([[np.sum([convolute1d(ts_layer, filter_layer, self.stride)
                                    for ts_layer, filter_layer in zip(_ts, _filter)], axis=0)
                             for _filter in self.weights]
                           for _ts in t])

        if self.bias is not None:
            return np.array([[layer + bias for layer, bias in zip(_ts, self.bias)] for _ts in result])
        else:
            return result


def convolute1d(ts, filter_matrix, stride):
    x_d = len(ts)
    x_f = len(filter_matrix)

    x_stride = stride[0]

    x_o = ((x_d - x_f) // x_stride) + 1

    def get_submatrix(matrix, x):
        index_column = x * x_stride
        return matrix[index_column: index_column + x_f]

    return np.array(
        [np.sum(get_submatrix(ts, x) * filter_matrix) for x in range(0, x_o)])

