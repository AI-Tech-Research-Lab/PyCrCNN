import numpy as np


def encode_matrix(HE, matrix):
    """Encode a float nD-matrix in a PyPtxt nD-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : nD-np.array( dtype=float )
        matrix to be encoded

    Returns
    -------
    matrix
        nD-np.array( dtype=PyPtxt ) with encoded values
    """

    try:
        return np.array(list(map(HE.encodeFrac, matrix)))
    except TypeError:
        return np.array([encode_matrix(HE, m) for m in matrix])


def decode_matrix(HE, matrix):
    """Decode a PyPtxt nD-matrix in a float nD-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : nD-np.array( dtype=PyPtxt )
        matrix to be decoded

    Returns
    -------
    matrix
        nD-np.array( dtype=float ) with float values
    """
    try:
        return np.array(list(map(HE.decodeFrac, matrix)))
    except TypeError:
        return np.array([decode_matrix(HE, m) for m in matrix])


def encrypt_matrix(HE, matrix):
    """Encrypt a float nD-matrix in a PyCtxt nD-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : nD-np.array( dtype=float )
        matrix to be encrypted

    Returns
    -------
    matrix
        nD-np.array( dtype=PyCtxt ) with encrypted values
    """
    try:
        return np.array(list(map(HE.encryptFrac, matrix)))
    except TypeError:
        return np.array([encrypt_matrix(HE, m) for m in matrix])


def decrypt_matrix(HE, matrix):
    """Decrypt a PyCtxt nD matrix in a float nD matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : nD-np.array( dtype=PyCtxt )
        matrix to be decrypted

    Returns
    -------
    matrix
        nD-np.array( dtype=float ) with plain values
    """
    try:
        return np.array(list(map(HE.decryptFrac, matrix)))
    except TypeError:
        return np.array([decrypt_matrix(HE, m) for m in matrix])
