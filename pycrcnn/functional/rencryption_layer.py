from ..crypto import crypto as cr


class RencryptionLayer:
    """
    A class used to represent a layer which decrypts and then
    re-encrypts a 2D matrix.
    While this is not a layer you would see in a CNN, this
    operation has been put in a layer to maintain coherence.
    ...

    Attributes
    ----------
    HE : Pyfhel
        Pyfhel object, used to perform the rencryption

    Methods
    -------
    __init__(self, HE)
        Constructor of the layer.

    __call__(self, t)
        Re-encrypts a 2D matrix in the form
            [y, x]
        using the keys given by the HE object.
    """
    def __init__(self, HE):
        self.HE = HE

    def __call__(self, image):
        plain = cr.decrypt_matrix_2d(self.HE, image)
        return cr.encrypt_matrix_2d(self.HE, plain)
