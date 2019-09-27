import numpy as np

from Pyfhel import PyCtxt

from ..crypto import crypto as c


class LinearLayer:
    """
    A class used to represent a linear (fully connected) layer
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to encode weights and bias
    weights : np.array( dtype=PyPtxt )
        Weights of the layer, in form
        [out_features, in_features]
    bias : np.array( dtype=PyPtxt ), default=None
        Biases of the layer, 1-D array


    Methods
    -------
    __init__(self, HE, weights, bias=None)
        Constructor of the layer, bias is set to None if not provided.
    __call__(self, t)
        Execute che linear operation on a flattened input, t, in the form
            [n_images, in_features]
        using weights and biases of the layer.
    """

    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = c.encode_matrix_2x2(HE, weights)
        self.bias = bias
        if bias is not None:
            self.bias = c.encode_vector(HE, bias)

    def __call__(self, t):
        result = linear_multiply(self.HE, t, self.weights)
        if self.bias is not None:
            n_images = len(result)
            for i in range(0, n_images):
                result[i] = result[i] + self.bias
        return result


def linear_multiply(HE, vector, matrix):
    """Execute the linear multiply operation, given a batch of flattened input
    and a weight matrix.


    Parameters
    ----------
    HE : Pyfhel
        Pyfhel object
    vector: 2D-np.array( dtype=PyCtxt )
        flattened input in the form
        [n_images, in_features]
    matrix: 2D-np.array( dtype=PyPtxt )
        weight to use for multiplication

    Returns
    -------
    result : 2D-np.array( dtype=PtCtxt )
        Encrypted result of the linear multiplication, in the form
        [n_images, out_features]
    """
    n_images = len(vector)
    out_features = len(matrix)
    in_features = len(matrix[0])

    result = np.empty( (n_images, out_features), dtype=PyCtxt)
    for n_image in range(0, n_images):
        for i in range(0, out_features):
            sum = HE.encryptFrac(0)
            for j in range(0, in_features):
                partial_mul = vector[n_image][j] * matrix[i][j]
                HE.relinearize(partial_mul)
                sum = sum + partial_mul
            result[n_image][i] = sum


    return result
