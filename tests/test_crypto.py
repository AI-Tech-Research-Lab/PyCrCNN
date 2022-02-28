import pytest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.crypto import crypto


class TestCrypto:

    @pytest.fixture
    def HE(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        return HE

    def test_encode_vector(self, HE):
        vector = np.array([1, 2, 3])
        result = crypto.encode_matrix(HE, vector)
        assert HE.decodeFrac(result[0]) == 1
        assert HE.decodeFrac(result[1]) == 2
        assert HE.decodeFrac(result[2]) == 3

    def test_decode_vector(self, HE):
        vector = np.array([HE.encodeFrac(1),
                           HE.encodeFrac(2),
                           HE.encodeFrac(3)])

        result = crypto.decode_matrix(HE, vector)
        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3

    def test_encode_matrix_2d(self, HE):
        matrix = np.array([
            [1, 2],
            [3, 4]
        ])

        result = crypto.encode_matrix(HE, matrix)
        assert HE.decodeFrac(result[0][0]) == 1
        assert HE.decodeFrac(result[0][1]) == 2
        assert HE.decodeFrac(result[1][0]) == 3
        assert HE.decodeFrac(result[1][1]) == 4

    def test_decode_matrix_2d(self, HE):
        matrix = np.array([
            [HE.encodeFrac(1), HE.encodeFrac(2)],
            [HE.encodeFrac(3), HE.encodeFrac(4)]
        ])

        result = crypto.decode_matrix(HE, matrix)
        assert result[0][0] == 1
        assert result[0][1] == 2
        assert result[1][0] == 3
        assert result[1][1] == 4

    def test_encode_matrix(self, HE):
        matrix = np.array([[
                [
                    [1, 2],
                    [3, 4]
                ],
                [
                    [5, 6],
                    [7, 8]
                ]
            ]
            ,[
                [
                    [10, 20],
                    [30, 40]
                ],
                [
                    [50, 60],
                    [70, 80]
                ]
            ]
        ])

        result = crypto.encode_matrix(HE, matrix)
        assert HE.decodeFrac(result[0][0][0][0]) == 1
        assert HE.decodeFrac(result[0][0][0][1]) == 2
        assert HE.decodeFrac(result[0][0][1][0]) == 3
        assert HE.decodeFrac(result[0][0][1][1]) == 4
        assert HE.decodeFrac(result[0][1][0][0]) == 5
        assert HE.decodeFrac(result[0][1][0][1]) == 6
        assert HE.decodeFrac(result[0][1][1][0]) == 7
        assert HE.decodeFrac(result[0][1][1][1]) == 8
        assert HE.decodeFrac(result[1][0][0][0]) == 10
        assert HE.decodeFrac(result[1][0][0][1]) == 20
        assert HE.decodeFrac(result[1][0][1][0]) == 30
        assert HE.decodeFrac(result[1][0][1][1]) == 40
        assert HE.decodeFrac(result[1][1][0][0]) == 50
        assert HE.decodeFrac(result[1][1][0][1]) == 60
        assert HE.decodeFrac(result[1][1][1][0]) == 70
        assert HE.decodeFrac(result[1][1][1][1]) == 80

    def test_decode_matrix(self, HE):
        matrix = np.array([
            [
                [
                    [HE.encodeFrac(1), HE.encodeFrac(2)]
                    , [HE.encodeFrac(3), HE.encodeFrac(4)]
                ],
                [
                    [HE.encodeFrac(5), HE.encodeFrac(6)]
                    , [HE.encodeFrac(7), HE.encodeFrac(8)]
                ]
            ]
            , [
                [
                    [HE.encodeFrac(10), HE.encodeFrac(20)]
                    , [HE.encodeFrac(30), HE.encodeFrac(40)]
                ],
                [
                    [HE.encodeFrac(50), HE.encodeFrac(60)]
                    , [HE.encodeFrac(70), HE.encodeFrac(80)]
                ]
            ]
        ])

        result = crypto.decode_matrix(HE, matrix)
        assert result[0][0][0][0] == 1
        assert result[0][0][0][1] == 2
        assert result[0][0][1][0] == 3
        assert result[0][0][1][1] == 4
        assert result[0][1][0][0] == 5
        assert result[0][1][0][1] == 6
        assert result[0][1][1][0] == 7
        assert result[0][1][1][1] == 8
        assert result[1][0][0][0] == 10
        assert result[1][0][0][1] == 20
        assert result[1][0][1][0] == 30
        assert result[1][0][1][1] == 40
        assert result[1][1][0][0] == 50
        assert result[1][1][0][1] == 60
        assert result[1][1][1][0] == 70
        assert result[1][1][1][1] == 80

    def test_encrypt_matrix(self, HE):
        matrix = np.array([[
            [
                [1, 2],
                [3, 4]
             ],
            [
                [5, 6],
                [7, 8]
             ]
            ]
            ,[
                [
                    [10, 20],
                    [30, 40]
                ],
                [
                    [50, 60],
                    [70, 80]
                ]
        ]])
        result = crypto.encrypt_matrix(HE, matrix)
        assert HE.decryptFrac(result[0][0][0][0]) == 1
        assert HE.decryptFrac(result[0][0][0][1]) == 2
        assert HE.decryptFrac(result[0][0][1][0]) == 3
        assert HE.decryptFrac(result[0][0][1][1]) == 4
        assert HE.decryptFrac(result[0][1][0][0]) == 5
        assert HE.decryptFrac(result[0][1][0][1]) == 6
        assert HE.decryptFrac(result[0][1][1][0]) == 7
        assert HE.decryptFrac(result[0][1][1][1]) == 8
        assert HE.decryptFrac(result[1][0][0][0]) == 10
        assert HE.decryptFrac(result[1][0][0][1]) == 20
        assert HE.decryptFrac(result[1][0][1][0]) == 30
        assert HE.decryptFrac(result[1][0][1][1]) == 40
        assert HE.decryptFrac(result[1][1][0][0]) == 50
        assert HE.decryptFrac(result[1][1][0][1]) == 60
        assert HE.decryptFrac(result[1][1][1][0]) == 70
        assert HE.decryptFrac(result[1][1][1][1]) == 80

    def test_encrypt_matrix_2d(self, HE):
        matrix = np.array([[1, 2],
                           [3, 4]])

        result = crypto.encrypt_matrix(HE, matrix)
        assert HE.decryptFrac(result[0][0]) == 1
        assert HE.decryptFrac(result[0][1]) == 2
        assert HE.decryptFrac(result[1][0]) == 3
        assert HE.decryptFrac(result[1][1]) == 4

    def test_decrypt_matrix_2d(self, HE):
        matrix = np.array([[1, 2]
                         , [3, 4]])
        encrypted_matrix = crypto.encrypt_matrix(HE, matrix)
        result = crypto.decrypt_matrix(HE, encrypted_matrix)
        assert np.allclose(result, matrix)

    def test_decrypt_matrix(self, HE):
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
        assert result[0][0][0][0] == 1
        assert result[0][0][0][1] == 2
        assert result[0][0][1][0] == 3
        assert result[0][0][1][1] == 4
        assert result[0][1][0][0] == 5
        assert result[0][1][0][1] == 6
        assert result[0][1][1][0] == 7
        assert result[0][1][1][1] == 8
        assert result[1][0][0][0] == 10
        assert result[1][0][0][1] == 20
        assert result[1][0][1][0] == 30
        assert result[1][0][1][1] == 40
        assert result[1][1][0][0] == 50
        assert result[1][1][0][1] == 60
        assert result[1][1][1][0] == 70
        assert result[1][1][1][1] == 80
