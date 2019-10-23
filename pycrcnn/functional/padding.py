import numpy as np


def apply_padding(t, padding):
    """Execute a padding operation given a batch of images in the form
        [n_image, n_layer, y, x]
       After the execution, the result will be in the form
        [n_image, n_layer, y+padding, x+padding]
       The element in the new rows/column will be zero.
       Due to Pyfhel limits, a sum/product between two PyPtxt can't be done.
       This leads to the need of having a PyCtxt which has to be zero if decrypted: this is done by
       subtracting an arbitrary value to itself.

    Parameters
    ----------
    t: np.array( dtype=PyCtxt )
        Encrypted image to execute the padding on, in the form
        [n_images, n_layer, y, x]
    padding: (int, int)

    Returns
    -------
    result : np.array( dtype=PyCtxt )
        Encrypted result of the padding, in the form
        [n_images, n_layer, y+padding, x+padding]
    """

    y_p = padding[0]
    x_p = padding[1]
    zero = t[0][0][y_p+1][x_p+1] - t[0][0][y_p+1][x_p+1]
    return [[np.pad(mat, ((y_p, y_p), (x_p, x_p)), 'constant', constant_values=zero) for mat in layer] for layer in t]
