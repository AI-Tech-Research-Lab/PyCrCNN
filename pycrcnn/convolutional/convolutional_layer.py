import numpy as np

from Pyfhel import PyCtxt

from pycrcnn.crypto import crypto as c


class ConvolutionalLayer:
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
    x_stride : int
        Horizontal stride
    y_stride : int
        Vertical stride
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

    def __init__(self, HE, weights, x_stride, y_stride, bias=None):
        self.HE = HE
        self.weights = c.encode_matrix(HE, weights)
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.bias = bias
        if bias is not None:
            self.bias = c.encode_vector(HE, bias)

    def __call__(self, t):
        result = convolute(self.HE, t, self.weights, self.x_stride, self.y_stride)
        if self.bias is not None:
            n_images = len(result)
            n_layers = len(result[0])
            for i in range(0, n_images):
                for j in range(0, n_layers):
                    result[i][j] = result[i][j] + self.bias[j]
        return result


def convolute(HE, image, filters, x_stride, y_stride):
    """Execute the convolution operation, given a batch of images,
    a set of weights and related strides.


    Parameters
    ----------
    HE : Pyfhel
        Pyfhel object
    image : np.array( dtype=PyCtxt )
        Encrypted image to execute the convolution on, in the form
        [n_images, n_layers, y, x]
    filters : np.array( dtype=PyPtxt )
        Encoded weights to use in the convolution, in the form
        [n_filter, n_layers, y, x]
    x_stride : int
        Horizontal stride
    y_stride : int
        Vertical stride

    Returns
    -------
    result : np.array( dtype=PtCtxt )
        Encrypted result of the convolution, in the form
        [n_images, n_layers, y, x]
    """
    n_images = len(image)
    n_layers = len(image[0])
    x_d = len(image[0][0])
    y_d = len(image[0][0][0])

    n_filters = len(filters)
    x_f = len(filters[0][0])
    y_f = len(filters[0][0][0])

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    result = np.empty((n_images, n_filters, x_o, y_o), dtype=PyCtxt)

    for n_image in range(0, n_images):

        for n_filter in range(0, n_filters):

            partial_result = c.encrypt_matrix_2x2(HE, np.zeros((x_o, y_o), dtype=float))
            for n_layer in range(0, n_layers):
                partial_result = partial_result + convolute2d(image[n_image][n_layer]
                                                              , filters[n_filter][n_layer]
                                                              , x_stride
                                                              , y_stride)
            result[n_image][n_filter] = partial_result

    return result


def convolute2d(image, filter_matrix, x_stride, y_stride):
    """Execute a convolution operation given an 2D-image, a 2D-filter
    and related strides.


    Parameters
    ----------
    image : np.array( dtype=PyCtxt )
        Encrypted image to execute the convolution on, in the form
        [y, x]
    filters : np.array( dtype=PyPtxt )
        Encoded weights to use in the convolution, in the form
        [y, x]
    x_stride : int
        Horizontal stride
    y_stride : int
        Vertical stride

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

    x_o = ((x_d - x_f) // x_stride) + 1
    y_o = ((y_d - y_f) // y_stride) + 1

    result = np.empty((x_o, y_o), dtype=PyCtxt)

    for i in range(0, y_o):
        index_row = i * y_stride
        for j in range(0, x_o):
            index_column = j * x_stride
            result[i][j] = np.sum(image[index_row:index_row + y_f
                                  , index_column:index_column + x_f] * filter_matrix)

    return result
