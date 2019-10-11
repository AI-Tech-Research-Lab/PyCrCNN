import numpy as np

from Pyfhel import PyCtxt


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

    def __call__(self, image):
        return avg_pool2d(self.HE, image, self.kernel_size, self.stride)


def avg_pool2d(HE, image, kernel_size, stride):
    """Execute the average pooling operation, given a batch of images,
        a kernel dimension and the stride.


        Parameters
        ----------
        HE : Pyfhel
            Pyfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted image to execute the convolution on, in the form
            [n_images, n_layers, y, x]
        kernel_size : int
            size of the square kernel
        stride : int

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the pooling, in the form
            [n_images, n_layers, y, x]
        """

    n_images = len(image)
    n_layers = len(image[0])
    x_d = len(image[0][0])
    y_d = len(image[0][0][0])

    x_o = ((x_d - kernel_size) // stride) + 1
    y_o = ((y_d - kernel_size) // stride) + 1

    result = np.empty((n_images, n_layers, x_o, y_o), dtype=PyCtxt)

    for n_image in range(0, n_images):
        for n_layer in range(0, n_layers):
            result[n_image][n_layer] = _avg(HE, image[n_image][n_layer], kernel_size, stride)
    return result


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

    result = np.empty((x_o, y_o), dtype=PyCtxt)

    for i in range(0, y_o):
        index_row = i * stride
        for j in range(0, x_o):
            index_column = j * stride
            result[i][j] = np.sum(image[index_row:index_row + kernel_size
                                  , index_column:index_column + kernel_size]) * denominator
            HE.relinearize(result[i][j])

    return result
