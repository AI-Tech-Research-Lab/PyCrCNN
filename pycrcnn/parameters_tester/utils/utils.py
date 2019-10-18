import numpy as np

from pycrcnn.crypto import crypto as cr


def get_max_error(HE, p_matrix, c_matrix):
    """Given a plain matrix and a cipher matrix, returns the max
    error between the two (the max delta) and the position in which
    this happens.

    Parameters
    ----------
    HE: Pyfhel
        Pyfhel object, needed to decrypt the cipher matrix
    p_matrix: np.array( dtype=float )
        Numpy array of plain values

    c_matrix: np.array( dtype=PyCtxt )
        Numpy array of encrypted values

    Returns
    -------
    max_error
    position: np.array(dtype = int)
        Position in the matrix in which the max error is found
    """
    dec_matrix = cr.decrypt_matrix(HE, c_matrix)

    max_error = np.max(abs(p_matrix - dec_matrix))
    position = np.unravel_index(np.argmax(abs(p_matrix - dec_matrix)), p_matrix.shape)

    return max_error, position


def get_min_noise(HE, matrix):
    """Returns the minimum noise budget found in the entire cipher matrix

    Parameters
    ----------
    HE: Pyfhel
        Pyfhel object
    matrix: np.array( dtype=PyCtxt )
        Encrypted matrix, either in the form

    Returns
    -------
    min_noise: int
    """

    def local(loc_matrix):
        try:
            return np.array(list(map(HE.noiseLevel, loc_matrix)))
        except TypeError:
            return np.array([local(m) for m in loc_matrix])

    return np.min(local(matrix))