import numpy as np

from Pyfhel import PyCtxt


def square(HE, image):
    """Execute the square operation, given a batch of images,

        Parameters
        ----------
        HE : Pyfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted image to execute the convolution on, in the form
            [n_images, n_layers, y, x]

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the square operation, in the form
            [n_images, n_layers, y, x]
        """

    n_images = len(image)
    n_layers = len(image[0])
    x_d = len(image[0][0])
    y_d = len(image[0][0][0])

    for n_image in range(0, n_images):
        for n_layer in range(0, n_layers):
            for x in range(0, x_d):
                for y in range(0, y_d):
                    image[n_image][n_layer][y][x] = HE.square(image[n_image][n_layer][y][x])
                    HE.relinearize(image[n_image][n_layer][y][x])
    return image


def square2d(HE, image):
    """Execute the square operation, given a 2D-matrix.

        Parameters
        ----------
        HE : Pyfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted image to execute the convolution on, in the form
            [y, x]

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the square operation, in the form
            [y, x]
        """

    y_d = len(image)
    x_d = len(image[0])

    for x in range(0, x_d):
        for y in range(0, y_d):
            image[y][x] = HE.square(image[y][x])
            HE.relinearize(image[y][x])
    return image
