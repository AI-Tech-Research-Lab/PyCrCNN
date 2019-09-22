import numpy as np
from Pyfhel import PyCtxt, PyPtxt


def encode_vector(HE, vector):
    """Encode a single vector in a PyPtxt vector.

    Parameters
    ----------
    HE : Pyfhel object
    vector : np.array( dtype=float )
        vector to be encoded

    Returns
    -------
    vector
        np.array( dtype=PyPtxt ) with encoded values
    """
    result = np.empty((len(vector)), dtype=PyPtxt)

    for i in range(0, len(vector)):
        result[i] = HE.encodeFrac(vector[i])
    return result


def encode_matrix_2x2(HE, matrix):
    """Encode a 2D-matrix in a PyPtxt 2D-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 2D-np.array( dtype=float )
        matrix to be encoded

    Returns
    -------
    matrix
        2D-np.array( dtype=PyPtxt ) with encoded values
    """
    n_rows = len(matrix)
    n_columns = len(matrix[0])
    result = np.empty((n_rows, n_columns), dtype=PyPtxt)
    for i in range(0, len(matrix)):
        for k in range(0, len(matrix[i])):
            result[i][k] = HE.encodeFrac(matrix[i][k])
    return result


def encode_matrix(HE, matrix):
    """Encode a 4D-matrix in a PyPtxt 4D-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 4D-np.array( dtype=float )
        matrix to be encoded

    Returns
    -------
    matrix
        4D-np.array( dtype=PyPtxt ) with encoded values
    """
    n_matrixes = len(matrix)
    n_layers = len(matrix[0])
    n_rows = len(matrix[0][0])
    n_columns = len(matrix[0][0][0])
    result = np.empty((n_matrixes, n_layers, n_rows, n_columns), dtype=PyPtxt)

    for n_matrix in range(0, n_matrixes):
        for n_layer in range(0, n_layers):
            for i in range(0, n_rows):
                for k in range(0, n_columns):
                    result[n_matrix][n_layer][i][k] = HE.encodeFrac(
                        matrix[n_matrix][n_layer][i][k]
                    )
    return result


def encrypt_matrix_2x2(HE, matrix):
    """Encrypt a 2x2 matrix in a PyCtxt 2x2 matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 2x2-np.array( dtype=float )
        matrix to be encrypted

    Returns
    -------
    matrix
        2x2-np.array( dtype=PyCtxt ) with encrypted values
    """
    n_rows = len(matrix)
    n_columns = len(matrix[0])
    result = np.empty((n_rows, n_columns), dtype=PyCtxt)
    for i in range(0, len(matrix)):
        for k in range(0, len(matrix[i])):
            result[i][k] = HE.encryptFrac(matrix[i][k])
    return result


def encrypt_matrix(HE, matrix):
    """Encrypt a 4D-matrix in a PyCtxt 4D-matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 4D-np.array( dtype=float )
        matrix to be encrypted

    Returns
    -------
    matrix
        4D-np.array( dtype=PyCtxt ) with encrypted values
    """
    n_matrixes = len(matrix)
    n_layers = len(matrix[0])
    n_rows = len(matrix[0][0])
    n_columns = len(matrix[0][0][0])
    result = np.empty((n_matrixes, n_layers, n_rows, n_columns), dtype=PyCtxt)

    for n_matrix in range(0, n_matrixes):
        for n_layer in range(0, n_layers):
            for i in range(0, n_rows):
                for k in range(0, n_columns):
                    result[n_matrix][n_layer][i][k] = HE.encryptFrac(
                        matrix[n_matrix][n_layer][i][k]
                    )
    return result


def decrypt_matrix_2x2(HE, matrix):
    """Decrypt a 2x2 matrix in a float 2x2 matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 2x2-np.array( dtype=PyCtxt )
        matrix to be decrypted

    Returns
    -------
    matrix
        2x2-np.array( dtype=float ) with decrypted values
    """
    n_rows = len(matrix)
    n_columns = len(matrix[0])
    result = np.empty((n_rows, n_columns), dtype=float)
    for i in range(0, len(matrix)):
        for k in range(0, len(matrix[i])):
            result[i][k] = HE.decryptFrac(matrix[i][k])
    return result


def decrypt_matrix(HE, matrix):
    """Decrypt a 4D matrix in a float 4D matrix.

    Parameters
    ----------
    HE : Pyfhel object
    matrix : 4D-np.array( dtype=PyCtxt )
        matrix to be decrypted

    Returns
    -------
    matrix
        4D-np.array( dtype=float ) with plain values
    """
    n_matrixes = len(matrix)
    n_layers = len(matrix[0])
    n_rows = len(matrix[0][0])
    n_columns = len(matrix[0][0][0])
    result = np.empty((n_matrixes, n_layers, n_rows, n_columns), dtype=float)

    for n_matrix in range(0, n_matrixes):
        for n_layer in range(0, n_layers):
            for i in range(0, n_rows):
                for k in range(0, n_columns):
                    result[n_matrix][n_layer][i][k] = HE.decryptFrac(
                        matrix[n_matrix][n_layer][i][k]
                    )
    return result
