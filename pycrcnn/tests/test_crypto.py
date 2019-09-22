import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.crypto import crypto


class TestSum(unittest.TestCase):

    def test_encode_vector(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        vector = np.array([1, 2, 3])
        result = crypto.encode_vector(HE, vector)
        self.assertEqual(HE.decodeFrac(result[0]), 1)
        self.assertEqual(HE.decodeFrac(result[1]), 2)
        self.assertEqual(HE.decodeFrac(result[2]), 3)

    def test_encode_matrix(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([
            [
                [
                    [1, 2]
                   ,[3, 4]
                ],
                [
                    [5, 6]
                   ,[7, 8]
                ]
            ]
           ,[
                [
                    [10, 20]
                   ,[30, 40]
                ],
                [
                    [50, 60]
                   ,[70, 80]
                 ]
            ]
        ])

        result = crypto.encode_matrix(HE, matrix)
        self.assertEqual(HE.decodeFrac(result[0][0][0][0]), 1)
        self.assertEqual(HE.decodeFrac(result[0][0][0][1]), 2)
        self.assertEqual(HE.decodeFrac(result[0][0][1][0]), 3)
        self.assertEqual(HE.decodeFrac(result[0][0][1][1]), 4)
        self.assertEqual(HE.decodeFrac(result[0][1][0][0]), 5)
        self.assertEqual(HE.decodeFrac(result[0][1][0][1]), 6)
        self.assertEqual(HE.decodeFrac(result[0][1][1][0]), 7)
        self.assertEqual(HE.decodeFrac(result[0][1][1][1]), 8)
        self.assertEqual(HE.decodeFrac(result[1][0][0][0]), 10)
        self.assertEqual(HE.decodeFrac(result[1][0][0][1]), 20)
        self.assertEqual(HE.decodeFrac(result[1][0][1][0]), 30)
        self.assertEqual(HE.decodeFrac(result[1][0][1][1]), 40)
        self.assertEqual(HE.decodeFrac(result[1][1][0][0]), 50)
        self.assertEqual(HE.decodeFrac(result[1][1][0][1]), 60)
        self.assertEqual(HE.decodeFrac(result[1][1][1][0]), 70)
        self.assertEqual(HE.decodeFrac(result[1][1][1][1]), 80)

    def test_encode_matrix_2x2(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([
            [1, 2]
           ,[3, 4]
        ])

        result = crypto.encode_matrix_2x2(HE, matrix)
        self.assertEqual(HE.decodeFrac(result[0][0]), 1)
        self.assertEqual(HE.decodeFrac(result[0][1]), 2)
        self.assertEqual(HE.decodeFrac(result[1][0]), 3)
        self.assertEqual(HE.decodeFrac(result[1][1]), 4)

    def test_encrypt_matrix(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[
            [[1, 2]
                , [3, 4]
             ],
            [[5, 6]
                , [7, 8]
             ]
        ]
            , [
                [[10, 20]
                    , [30, 40]
                 ],
                [[50, 60]
                    , [70, 80]
                 ]
            ]])
        result = crypto.encrypt_matrix(HE, matrix)
        self.assertEqual(HE.decryptFrac(result[0][0][0][0]), 1)
        self.assertEqual(HE.decryptFrac(result[0][0][0][1]), 2)
        self.assertEqual(HE.decryptFrac(result[0][0][1][0]), 3)
        self.assertEqual(HE.decryptFrac(result[0][0][1][1]), 4)
        self.assertEqual(HE.decryptFrac(result[0][1][0][0]), 5)
        self.assertEqual(HE.decryptFrac(result[0][1][0][1]), 6)
        self.assertEqual(HE.decryptFrac(result[0][1][1][0]), 7)
        self.assertEqual(HE.decryptFrac(result[0][1][1][1]), 8)
        self.assertEqual(HE.decryptFrac(result[1][0][0][0]), 10)
        self.assertEqual(HE.decryptFrac(result[1][0][0][1]), 20)
        self.assertEqual(HE.decryptFrac(result[1][0][1][0]), 30)
        self.assertEqual(HE.decryptFrac(result[1][0][1][1]), 40)
        self.assertEqual(HE.decryptFrac(result[1][1][0][0]), 50)
        self.assertEqual(HE.decryptFrac(result[1][1][0][1]), 60)
        self.assertEqual(HE.decryptFrac(result[1][1][1][0]), 70)
        self.assertEqual(HE.decryptFrac(result[1][1][1][1]), 80)

    def test_encrypt_matrix_2x2(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[1, 2]
                          ,[3, 4]])
        result = crypto.encrypt_matrix_2x2(HE, matrix)
        self.assertEqual(HE.decryptFrac(result[0][0]), 1)
        self.assertEqual(HE.decryptFrac(result[0][1]), 2)
        self.assertEqual(HE.decryptFrac(result[1][0]), 3)
        self.assertEqual(HE.decryptFrac(result[1][1]), 4)

    def test_decrypt_matrix_2x2(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[1, 2]
                          ,[3, 4]])
        encrypted_matrix = crypto.encrypt_matrix_2x2(HE, matrix)
        result = crypto.decrypt_matrix_2x2(HE, encrypted_matrix)
        self.assertEqual(result.all(), matrix.all())

    def test_decrypt_matrix(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[
            [[1, 2]
                , [3, 4]
             ],
            [[5, 6]
                , [7, 8]
             ]
        ]
            , [
                [[10, 20]
                    , [30, 40]
                 ],
                [[50, 60]
                    , [70, 80]
                 ]
            ]])
        enc_matrix = crypto.encrypt_matrix(HE, matrix)
        result = crypto.decrypt_matrix(HE, enc_matrix)
        self.assertEqual(result[0][0][0][0], 1)
        self.assertEqual(result[0][0][0][1], 2)
        self.assertEqual(result[0][0][1][0], 3)
        self.assertEqual(result[0][0][1][1], 4)
        self.assertEqual(result[0][1][0][0], 5)
        self.assertEqual(result[0][1][0][1], 6)
        self.assertEqual(result[0][1][1][0], 7)
        self.assertEqual(result[0][1][1][1], 8)
        self.assertEqual(result[1][0][0][0], 10)
        self.assertEqual(result[1][0][0][1], 20)
        self.assertEqual(result[1][0][1][0], 30)
        self.assertEqual(result[1][0][1][1], 40)
        self.assertEqual(result[1][1][0][0], 50)
        self.assertEqual(result[1][1][0][1], 60)
        self.assertEqual(result[1][1][1][0], 70)
        self.assertEqual(result[1][1][1][1], 80)


if __name__ == '__main__':
    unittest.main()
