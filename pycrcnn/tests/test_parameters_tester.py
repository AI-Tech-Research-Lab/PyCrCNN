import unittest
import numpy as np
from Pyfhel import Pyfhel
from pycrcnn.parameters_tester.param_tester_cli import get_max_error, get_min_noise

from pycrcnn.crypto.crypto import encrypt_matrix_2d


class ParametersTesterCase(unittest.TestCase):

    def test_get_max_error(self):

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        plain_image = np.array([
            [1, 2, 3],
            [1, 3, 4],
            [1, 3, 4]
        ])

        cipher_image = encrypt_matrix_2d(HE, plain_image)

        cipher_image[2][0] = HE.encryptFrac(-1)

        max_error, position = get_max_error(HE, plain_image, cipher_image)

        self.assertEqual(max_error, 2)
        self.assertEqual(position, (2, 0))

    def test_get_min_noise(self):

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        plain_image = np.array([
            [1, 2, 3],
            [1, 3, 4],
            [1, 3, 4]
        ])

        cipher_image = encrypt_matrix_2d(HE, plain_image)
        cipher_image[1][1] = cipher_image[1][1] * HE.encryptFrac(2)

        self.assertEqual(get_min_noise(HE, cipher_image), HE.noiseLevel(cipher_image[1][1]))


if __name__ == '__main__':
    unittest.main()
