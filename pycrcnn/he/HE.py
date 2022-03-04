import numpy as np
from Pyfhel import Pyfhel


class HE:
    def encode_matrix(self, matrix):
        """Encode a matrix in a plaintext HE nD-matrix.

        Parameters
        ----------
        matrix : nD-np.array( dtype=float )
            matrix to be encoded

        Returns
        -------
        matrix
            nD-np.array with encoded values
        """
        pass

    def decode_matrix(self, matrix):
        pass

    def encrypt_matrix(self, matrix):
        pass

    def decrypt_matrix(self, matrix):
        pass


class BFVPyfhel(HE):
    def __init__(self, m, p, sec=128, int_digits=64, frac_digits=32):
        self.he = Pyfhel()
        self.he.contextGen(int_m=m, int_p=p, int_sec=sec, int_fracDigits=frac_digits,
                           int_intDigits=int_digits)
        self.he.keyGen()

    def encode_matrix(self, matrix):
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
            return np.array(list(map(self.he.encodeFrac, matrix)))
        except TypeError:
            return np.array([self.encode_matrix(m) for m in matrix])

    def decode_matrix(self, matrix):
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
            return np.array(list(map(self.he.decodeFrac, matrix)))
        except TypeError:
            return np.array([self.decode_matrix(m) for m in matrix])

    def encrypt_matrix(self, matrix):
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
            return np.array(list(map(self.he.encryptFrac, matrix)))
        except TypeError:
            return np.array([self.encrypt_matrix(m) for m in matrix])

    def decrypt_matrix(self, matrix):
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
            return np.array(list(map(self.he.decryptFrac, matrix)))
        except TypeError:
            return np.array([self.decrypt_matrix(m) for m in matrix])



