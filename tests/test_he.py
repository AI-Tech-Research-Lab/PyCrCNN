import pytest

import numpy as np
from pycrcnn.he.HE import BFVPyfhel, CKKSPyfhel, CKKSTenSEAL, MockHE


class TestCrypto:

    @pytest.fixture(params=[MockHE, BFVPyfhel, CKKSTenSEAL])
    # @pytest.fixture(params=[MockHE, BFVPyfhel, CKKSPyfhel, CKKSTenSEAL])
    def HE(self, request):
        HE = request.param()
        HE.generate_keys()
        return HE

    def test_encode_vector(self, HE):
        vector = np.array([1, 2, 3])
        result = HE.encode_matrix(vector)

        assert np.isclose(HE.decode(result[0]), 1)
        assert np.isclose(HE.decode(result[1]), 2)
        assert np.isclose(HE.decode(result[2]), 3)

    def test_decode_vector(self, HE):
        vector = np.array([HE.encode(1),
                           HE.encode(2),
                           HE.encode(3)])

        result = HE.decode_matrix(vector)
        assert np.isclose(result[0], 1)
        assert np.isclose(result[1], 2)
        assert np.isclose(result[2], 3)

    def test_encode_matrix_2d(self, HE):
        matrix = np.array([
            [1, 2],
            [3, 4]
        ])

        result = HE.encode_matrix(matrix)
        assert np.isclose(HE.decode(result[0][0]), 1)
        assert np.isclose(HE.decode(result[0][1]), 2)
        assert np.isclose(HE.decode(result[1][0]), 3)
        assert np.isclose(HE.decode(result[1][1]), 4)

    def test_decode_matrix_2d(self, HE):
        matrix = np.array([
            [HE.encode(1), HE.encode(2)],
            [HE.encode(3), HE.encode(4)]
        ])

        result = HE.decode_matrix(matrix)
        assert np.isclose(result[0][0], 1)
        assert np.isclose(result[0][1], 2)
        assert np.isclose(result[1][0], 3)
        assert np.isclose(result[1][1], 4)

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

        result = HE.encode_matrix(matrix)
        assert np.isclose(HE.decode(result[0][0][0][0]), 1)
        assert np.isclose(HE.decode(result[0][0][0][1]), 2)
        assert np.isclose(HE.decode(result[0][0][1][0]), 3)
        assert np.isclose(HE.decode(result[0][0][1][1]), 4)
        assert np.isclose(HE.decode(result[0][1][0][0]), 5)
        assert np.isclose(HE.decode(result[0][1][0][1]), 6)
        assert np.isclose(HE.decode(result[0][1][1][0]), 7)
        assert np.isclose(HE.decode(result[0][1][1][1]), 8)
        assert np.isclose(HE.decode(result[1][0][0][0]), 10)
        assert np.isclose(HE.decode(result[1][0][0][1]), 20)
        assert np.isclose(HE.decode(result[1][0][1][0]), 30)
        assert np.isclose(HE.decode(result[1][0][1][1]), 40)
        assert np.isclose(HE.decode(result[1][1][0][0]), 50)
        assert np.isclose(HE.decode(result[1][1][0][1]), 60)
        assert np.isclose(HE.decode(result[1][1][1][0]), 70)
        assert np.isclose(HE.decode(result[1][1][1][1]), 80)

    def test_decode_matrix(self, HE):
        matrix = np.array([
            [
                [
                    [HE.encode(1), HE.encode(2)]
                    , [HE.encode(3), HE.encode(4)]
                ],
                [
                    [HE.encode(5), HE.encode(6)]
                    , [HE.encode(7), HE.encode(8)]
                ]
            ]
            , [
                [
                    [HE.encode(10), HE.encode(20)]
                    , [HE.encode(30), HE.encode(40)]
                ],
                [
                    [HE.encode(50), HE.encode(60)]
                    , [HE.encode(70), HE.encode(80)]
                ]
            ]
        ])

        result = HE.decode_matrix(matrix)
        assert np.isclose(result[0][0][0][0], 1)
        assert np.isclose(result[0][0][0][1], 2)
        assert np.isclose(result[0][0][1][0], 3)
        assert np.isclose(result[0][0][1][1], 4)
        assert np.isclose(result[0][1][0][0], 5)
        assert np.isclose(result[0][1][0][1], 6)
        assert np.isclose(result[0][1][1][0], 7)
        assert np.isclose(result[0][1][1][1], 8)
        assert np.isclose(result[1][0][0][0], 10)
        assert np.isclose(result[1][0][0][1], 20)
        assert np.isclose(result[1][0][1][0], 30)
        assert np.isclose(result[1][0][1][1], 40)
        assert np.isclose(result[1][1][0][0], 50)
        assert np.isclose(result[1][1][0][1], 60)
        assert np.isclose(result[1][1][1][0], 70)
        assert np.isclose(result[1][1][1][1], 80)

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
        result = HE.encrypt_matrix(matrix)
        assert np.isclose(HE.decrypt(result[0][0][0][0]), 1)
        assert np.isclose(HE.decrypt(result[0][0][0][1]), 2)
        assert np.isclose(HE.decrypt(result[0][0][1][0]), 3)
        assert np.isclose(HE.decrypt(result[0][0][1][1]), 4)
        assert np.isclose(HE.decrypt(result[0][1][0][0]), 5)
        assert np.isclose(HE.decrypt(result[0][1][0][1]), 6)
        assert np.isclose(HE.decrypt(result[0][1][1][0]), 7)
        assert np.isclose(HE.decrypt(result[0][1][1][1]), 8)
        assert np.isclose(HE.decrypt(result[1][0][0][0]), 10)
        assert np.isclose(HE.decrypt(result[1][0][0][1]), 20)
        assert np.isclose(HE.decrypt(result[1][0][1][0]), 30)
        assert np.isclose(HE.decrypt(result[1][0][1][1]), 40)
        assert np.isclose(HE.decrypt(result[1][1][0][0]), 50)
        assert np.isclose(HE.decrypt(result[1][1][0][1]), 60)
        assert np.isclose(HE.decrypt(result[1][1][1][0]), 70)
        assert np.isclose(HE.decrypt(result[1][1][1][1]), 80)

    def test_encrypt_matrix_2d(self, HE):
        matrix = np.array([[1, 2],
                           [3, 4]])

        result = HE.encrypt_matrix(matrix)
        assert np.isclose(HE.decrypt(result[0][0]), 1)
        assert np.isclose(HE.decrypt(result[0][1]), 2)
        assert np.isclose(HE.decrypt(result[1][0]), 3)
        assert np.isclose(HE.decrypt(result[1][1]), 4)

    def test_decrypt_matrix_2d(self, HE):
        matrix = np.array([[1, 2]
                         , [3, 4]])
        encrypted_matrix = HE.encrypt_matrix(matrix)
        result = HE.decrypt_matrix(encrypted_matrix)
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
        enc_matrix = HE.encrypt_matrix(matrix)
        result = HE.decrypt_matrix(enc_matrix)
        assert np.isclose(result[0][0][0][0], 1, atol=0.1)
        assert np.isclose(result[0][0][0][1], 2, atol=0.1)
        assert np.isclose(result[0][0][1][0], 3, atol=0.1)
        assert np.isclose(result[0][0][1][1], 4, atol=0.1)
        assert np.isclose(result[0][1][0][0], 5, atol=0.1)
        assert np.isclose(result[0][1][0][1], 6, atol=0.1)
        assert np.isclose(result[0][1][1][0], 7, atol=0.1)
        assert np.isclose(result[0][1][1][1], 8, atol=0.1)
        assert np.isclose(result[1][0][0][0], 10, atol=0.1)
        assert np.isclose(result[1][0][0][1], 20, atol=0.1)
        assert np.isclose(result[1][0][1][0], 30, atol=0.1)
        assert np.isclose(result[1][0][1][1], 40, atol=0.1)
        assert np.isclose(result[1][1][0][0], 50, atol=0.1)
        assert np.isclose(result[1][1][0][1], 60, atol=0.1)
        assert np.isclose(result[1][1][1][0], 70, atol=0.1)
        assert np.isclose(result[1][1][1][1], 80, atol=0.1)
