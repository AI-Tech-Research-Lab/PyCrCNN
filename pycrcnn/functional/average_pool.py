import numpy as np

from Pyfhel import PyCtxt

from ..crypto import crypto as c


def avg_pool2d(HE, image, kernel_size, stride):
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

    return result
