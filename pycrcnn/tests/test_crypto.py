import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.crypto import crypto


class TestCrypto(unittest.TestCase):

    def test_encode_vector(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        vector = np.array([1, 2, 3])
        result = crypto.encode_vector(HE, vector)
        self.assertEqual(HE.decodeFrac(result[0]), 1)
        self.assertEqual(HE.decodeFrac(result[1]), 2)
        self.assertEqual(HE.decodeFrac(result[2]), 3)

    def test_decode_vector(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        vector = np.array([HE.encodeFrac(1),
                           HE.encodeFrac(2),
                           HE.encodeFrac(3)])

        result = crypto.decode_vector(HE,vector)
        self.assertEqual((result[0]), 1)
        self.assertEqual((result[1]), 2)
        self.assertEqual((result[2]), 3)

    def test_encode_matrix_2d(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([
            [1, 2]
           ,[3, 4]
        ])

        result = crypto.encode_matrix_2d(HE, matrix)
        self.assertEqual(HE.decodeFrac(result[0][0]), 1)
        self.assertEqual(HE.decodeFrac(result[0][1]), 2)
        self.assertEqual(HE.decodeFrac(result[1][0]), 3)
        self.assertEqual(HE.decodeFrac(result[1][1]), 4)

    def test_decode_matrix_2d(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([
            [HE.encodeFrac(1), HE.encodeFrac(2)]
           ,[HE.encodeFrac(3), HE.encodeFrac(4)]
        ])

        result = crypto.decode_matrix_2d(HE, matrix)
        self.assertEqual((result[0][0]), 1)
        self.assertEqual((result[0][1]), 2)
        self.assertEqual((result[1][0]), 3)
        self.assertEqual((result[1][1]), 4)

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

    def test_decode_matrix(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([
            [
                [
                    [HE.encodeFrac(1), HE.encodeFrac(2)]
                   ,[HE.encodeFrac(3), HE.encodeFrac(4)]
                ],
                [
                    [HE.encodeFrac(5), HE.encodeFrac(6)]
                   ,[HE.encodeFrac(7), HE.encodeFrac(8)]
                ]
            ]
           ,[
                [
                    [HE.encodeFrac(10), HE.encodeFrac(20)]
                   ,[HE.encodeFrac(30), HE.encodeFrac(40)]
                ],
                [
                    [HE.encodeFrac(50), HE.encodeFrac(60)]
                   ,[HE.encodeFrac(70), HE.encodeFrac(80)]
                 ]
            ]
        ])

        result = crypto.decode_matrix(HE, matrix)
        self.assertEqual((result[0][0][0][0]), 1)
        self.assertEqual((result[0][0][0][1]), 2)
        self.assertEqual((result[0][0][1][0]), 3)
        self.assertEqual((result[0][0][1][1]), 4)
        self.assertEqual((result[0][1][0][0]), 5)
        self.assertEqual((result[0][1][0][1]), 6)
        self.assertEqual((result[0][1][1][0]), 7)
        self.assertEqual((result[0][1][1][1]), 8)
        self.assertEqual((result[1][0][0][0]), 10)
        self.assertEqual((result[1][0][0][1]), 20)
        self.assertEqual((result[1][0][1][0]), 30)
        self.assertEqual((result[1][0][1][1]), 40)
        self.assertEqual((result[1][1][0][0]), 50)
        self.assertEqual((result[1][1][0][1]), 60)
        self.assertEqual((result[1][1][1][0]), 70)
        self.assertEqual((result[1][1][1][1]), 80)

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

    def test_encrypt_matrix_2d(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[1, 2]
                          ,[3, 4]])
        result = crypto.encrypt_matrix_2d(HE, matrix)
        self.assertEqual(HE.decryptFrac(result[0][0]), 1)
        self.assertEqual(HE.decryptFrac(result[0][1]), 2)
        self.assertEqual(HE.decryptFrac(result[1][0]), 3)
        self.assertEqual(HE.decryptFrac(result[1][1]), 4)

    def test_decrypt_matrix_2d(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        matrix = np.array([[1, 2]
                          ,[3, 4]])
        encrypted_matrix = crypto.encrypt_matrix_2d(HE, matrix)
        result = crypto.decrypt_matrix_2d(HE, encrypted_matrix)
        self.assertTrue(np.allclose(result, matrix))

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
