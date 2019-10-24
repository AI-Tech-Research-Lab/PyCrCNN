import numpy as np

from pycrcnn.crypto import crypto as cr
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer


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


def test_net(HE, net, encoded_layers, images, verbose):
    enc_images = cr.encrypt_matrix(HE, images.detach().numpy())
    net_iterator = net.children()

    def print_stats():
        max_error_value, max_error_position = get_max_error(HE, images.detach().numpy(), enc_images)
        print("\n------------ INTERMEDIATE STATS ------------------------")
        print("Max error = ", max_error_value, " at ",  max_error_position)
        print("Plain value = ", images[max_error_position].detach().numpy(), " , cipher value = "
              , HE.decryptFrac(enc_images[max_error_position]))
        print("Avg value= ", np.average(images.detach().numpy()))
        print("Min noise= ", get_min_noise(HE, enc_images))

    if verbose:
        print("Min noise initial= ", get_min_noise(HE, enc_images))

    for layer in encoded_layers:
        enc_images = layer(enc_images)
        if not (type(layer) == RencryptionLayer):
            images = next(net_iterator)(images)

        if verbose and not (type(layer) == FlattenLayer) and not (type(layer) == RencryptionLayer):
            print_stats()

    dec_matrix = cr.decrypt_matrix(HE, enc_images)
    final_error = np.max(abs(images.detach().numpy() - dec_matrix))
    if verbose:
        print("\n------------ FINAL RESULTS --------------------------")
        print(images)
        print(dec_matrix)

    return final_error