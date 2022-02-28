import pytest
import numpy as np
from Pyfhel import Pyfhel
from pycrcnn.parameters_tester.utils.utils import get_max_error, get_min_noise

from pycrcnn.crypto.crypto import encrypt_matrix


class TestsParametersTester:

    def test_get_max_error(self):

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        plain_image = np.array([
            [1, 2, 3],
            [1, 3, 4],
            [1, 3, 4]
        ])

        cipher_image = encrypt_matrix(HE, plain_image)

        cipher_image[2][0] = HE.encryptFrac(-1)

        max_error, position = get_max_error(HE, plain_image, cipher_image)

        assert max_error == 2
        assert position, (2 == 0)

    def test_get_min_noise(self):

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        plain_image = np.array([[
            [1, 2, 3],
            [1, 3, 4],
            [1, 3, 4]
        ]])

        cipher_image = encrypt_matrix(HE, plain_image)
        cipher_image[0][1][1] = cipher_image[0][1][1] * HE.encryptFrac(2)

        assert get_min_noise(HE, cipher_image) == HE.noiseLevel(cipher_image[0][1][1])

