import numpy as np

from pycrcnn.functional.padding import apply_padding


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
    def __init__(self, HE, kernel_size, stride=(1, 1), padding=(0, 0)):
        self.HE = HE
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, t):
        t = apply_padding(t, self.padding)
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
        kernel_size : (int, int)
            size of the kernel (y, x)
        stride : (int, int)
            stride (y, x)
        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the pooling, in the form
            [y, x]
        """
    x_s = stride[1]
    y_s = stride[0]

    x_k = kernel_size[1]
    y_k = kernel_size[0]

    x_d = len(image[0])
    y_d = len(image)

    x_o = ((x_d - x_k) // x_s) + 1
    y_o = ((y_d - y_k) // y_s) + 1

    denominator = HE.encode_number(1 / (x_k * y_k))

    def get_submatrix(matrix, x, y):
        index_row = y * y_s
        index_column = x * x_s
        return matrix[index_row: index_row + y_k, index_column: index_column + x_k]

    return [[np.sum(get_submatrix(image, x, y)) * denominator for x in range(0, x_o)] for y in range(0, y_o)]
