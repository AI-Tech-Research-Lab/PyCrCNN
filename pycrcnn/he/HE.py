import numpy as np
from Pyfhel import Pyfhel

import tempfile

tmp_dir = tempfile.TemporaryDirectory()


class HE:
    def generate_keys(self):
        pass

    def generate_relin_keys(self):
        pass

    def get_public_key(self):
        pass

    def get_relin_key(self):
        pass

    def load_public_key(self, key):
        pass

    def load_relin_key(self, key):
        pass

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

    def encode_number(self, number):
        pass

    def power(self, number, exp):
        pass

    def noise_budget(self, ciphertext):
        pass


class BFVPyfhel(HE):
    def __init__(self, m, p, sec=128, int_digits=64, frac_digits=32):
        self.he = Pyfhel()
        self.he.contextGen(p, m=m, sec=sec, fracDigits=frac_digits,
                           intDigits=int_digits)

    def generate_keys(self):
        self.he.keyGen()

    def generate_relin_keys(self, bitCount=60, size=3):
        self.he.relinKeyGen(bitCount, size)

    def get_public_key(self):
        self.he.savepublicKey(tmp_dir.name + "/pub.key")
        with open(tmp_dir.name + "/pub.key", 'rb') as f:
            return f.read()

    def get_relin_key(self):
        self.he.saverelinKey(tmp_dir.name + "/relin.key")
        with open(tmp_dir.name + "/relin.key", 'rb') as f:
            return f.read()

    def load_public_key(self, key):
        with open(tmp_dir.name + "/pub.key", 'wb') as f:
            f.write(key)
        self.he.restorepublicKey(tmp_dir.name + "/pub.key")

    def load_relin_key(self, key):
        with open(tmp_dir.name + "/relin.key", 'wb') as f:
            f.write(key)
        self.he.restorerelinKey(tmp_dir.name + "/relin.key")

    def encode_matrix(self, matrix):
        """Encode a float nD-matrix in a PyPtxt nD-matrix.

        Parameters
        ----------
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

    def encode_number(self, number):
        return self.he.encode(number)

    def power(self, number, exp):
        return self.he.power(number, exp)

    def noise_budget(self, ciphertext):
        try:
            return self.he.noiseLevel(ciphertext)
        except SystemError:
            return "Can't get NB without secret key."

