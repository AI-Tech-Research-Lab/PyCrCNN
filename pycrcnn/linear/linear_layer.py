import numpy as np

from ..crypto import crypto as c
from ..he.HE import CKKSPyfhel


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
            [n_images, in_features], 2D-np.array( dtype=PyCtxt )
        using weights and biases of the layer.
        returns the result in the form
            [n_images, out_features], 2D-np.array( dtype=PtCtxt )
    """

    def __init__(self, HE, weights, bias=None):
        self.HE = HE
        self.weights = HE.encode_matrix(weights)
        self.bias = bias
        if bias is not None:
            self.bias = HE.encode_matrix(bias)

    def __call__(self, t):
        if isinstance(self.HE, CKKSPyfhel):
            for w in np.ravel(self.weights):
                for i in range(0, t[0][0].mod_level):
                    self.HE.he.mod_switch_to_next(w)

        result = np.array([[np.sum(image * row) for row in self.weights] for image in t])
        if self.bias is not None:
            result = np.array([row + self.bias for row in result])
        return result

