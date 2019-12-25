import numpy as np


class SquareLayer:
    """
    A class used to represent a layer which performs the square
    of the values in a nD-matrix.

    ...

    Notes
    -----
    The values inside the matrix are squared, not the matrix themself

    Attributes
    ----------

    HE: Pyfhel
        Pyfhel object

    Methods
    -------
    __init__(self, HE)
        Constructor of the layer.
    __call__(self, image)
        Executes the square of the input matrix.
    """
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        return square(self.HE, image)


def square(HE, image):
    """Execute the square operation, given a batch of images,

        Parameters
        ----------
        HE : Pyfhel object
        image : np.array( dtype=PyCtxt )
            Encrypted nD matrix to square

        Returns
        -------
        result : np.array( dtype=PtCtxt )
            Encrypted result of the square operation
        """

    try:
        return np.array(list(map(lambda x: HE.power(x, 2), image)))
    except TypeError:
        return np.array([square(HE, m) for m in image])
