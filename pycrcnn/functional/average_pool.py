import numpy as np


class AveragePoolLayer:
    """
    A class used to represent a layer which performs an average pool operation
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to perform the pool operation
    kernel_size: int
        Size of the square kernel
    stride : int
        Stride of the pool operaiton

    Methods
    -------
    __init__(self, HE, kernel_size, stride)
        Constructor of the layer.
    __call__(self, t)
        Execute che average pooling operation on a batch of images, t, in the form
            [n_images, n_layers, y, x]
        using kernel size and strides of the layer.
    """
    def __init__(self, HE, kernel_size, stride):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, t):
        return np.array([[_avg(self.HE, layer, self.kernel_size, self.stride) for layer in image] for image in t])


def _avg(HE, image, kernel_size, stride):
    """Execute an average pooling operation given an 2D-image,
        a kernel-size and a stride.


        Parameters
        ----------
        HE: PYfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted image to execute the pooling, in the form
            [y, x]
        kernel_size : int
            size of the square kernel
        stride : int

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the pooling, in the form
            [y, x]
        """
    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - kernel_size) // stride) + 1
    y_o = ((y_d - kernel_size) // stride) + 1

    denominator = HE.encodeFrac(1 / (kernel_size * kernel_size))

    def get_submatrix(matrix, x, y):
        index_row = y * stride
        index_column = x * stride
        return matrix[index_row: index_row + kernel_size, index_column: index_column + kernel_size]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]
