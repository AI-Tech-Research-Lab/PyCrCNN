import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.functional import average_pool as avg
from pycrcnn.functional import square_layer as sq
from pycrcnn.crypto import crypto as cr
from pycrcnn.functional.average_pool import AveragePoolLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.functional.flatten_layer import FlattenLayer


class AverageLayerTests(unittest.TestCase):

    def setUp(self) -> None:

        self.HE = Pyfhel()
        self.HE.contextGen(65537)
        self.HE.keyGen()
        self.HE.relinKeyGen(20, 100)

        self.image = np.array([
            [[[1, 1, -1, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [0, -2, 1, 1, 0],
              [0, 1, 1, 0, 0]],
             [[1, 0, 1, 1, 0],
              [2, 1, 2, 0, 0],
              [0, 0, -1, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 1, 0, 0, 0]]],
            [[[1, 1, 1, 0, 0],
              [0, 1, 2, 1, 1],
              [0, 0, 1, 1, 1],
              [1, 0, 2, 3, 0],
              [0, 0, 1, 0, 0]],
             [[1, 0, 2, 0, 0],
              [0, 1, 1, 1, 2],
              [1, 0, 0, 1, 1],
              [0, -3, 1, 1, 0],
              [0, 0, 2, 0, 0]]]])

    def test_avg_pool2d_1(self):

        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        avg_layer = AveragePoolLayer(self.HE, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0))

        result = cr.decrypt_matrix(self.HE, avg_layer(encrypted_image))

        expected_result = np.array(
            [[[[ 0.7500,  0.5000,  0.2500,  0.2500],
              [ 0.2500,  0.7500,  1.0000,  0.7500],
              [-0.5000,  0.0000,  1.0000,  0.7500],
              [-0.2500,  0.2500,  0.7500,  0.2500]],

             [[ 1.0000,  1.0000,  1.0000,  0.2500],
              [ 0.7500,  0.5000,  0.2500,  0.2500],
              [ 0.2500,  0.2500,  0.2500,  0.5000],
              [ 0.5000,  0.7500,  0.5000,  0.2500]]],


            [[[ 0.7500,  1.2500,  1.0000,  0.5000],
              [ 0.2500,  1.0000,  1.2500,  1.0000],
              [ 0.2500,  0.7500,  1.7500,  1.2500],
              [ 0.2500,  0.7500,  1.5000,  0.7500]],

             [[ 0.5000,  1.0000,  1.0000,  0.7500],
              [ 0.5000,  0.5000,  0.7500,  1.2500],
              [-0.5000, -0.5000,  0.7500,  0.7500],
              [-0.7500,  0.0000,  1.0000,  0.2500]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_avg_pool2d_2(self):

        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        avg_layer = AveragePoolLayer(self.HE, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))

        result = cr.decrypt_matrix(self.HE, avg_layer(encrypted_image))

        expected_result = np.array(
            [[[[0.2500, 0.5000, 0.0000, -0.2500, 0.0000, 0.0000],
               [0.2500, 0.7500, 0.5000, 0.2500, 0.2500, 0.0000],
               [0.0000, 0.2500, 0.7500, 1.0000, 0.7500, 0.2500],
               [0.0000, -0.5000, 0.0000, 1.0000, 0.7500, 0.2500],
               [0.0000, -0.2500, 0.2500, 0.7500, 0.2500, 0.0000],
               [0.0000, 0.2500, 0.5000, 0.2500, 0.0000, 0.0000]],

              [[0.2500, 0.2500, 0.2500, 0.5000, 0.2500, 0.0000],
               [0.7500, 1.0000, 1.0000, 1.0000, 0.2500, 0.0000],
               [0.5000, 0.7500, 0.5000, 0.2500, 0.2500, 0.2500],
               [0.0000, 0.2500, 0.2500, 0.2500, 0.5000, 0.2500],
               [0.0000, 0.5000, 0.7500, 0.5000, 0.2500, 0.0000],
               [0.0000, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000]]],

             [[[0.2500, 0.5000, 0.5000, 0.2500, 0.0000, 0.0000],
               [0.2500, 0.7500, 1.2500, 1.0000, 0.5000, 0.2500],
               [0.0000, 0.2500, 1.0000, 1.2500, 1.0000, 0.5000],
               [0.2500, 0.2500, 0.7500, 1.7500, 1.2500, 0.2500],
               [0.2500, 0.2500, 0.7500, 1.5000, 0.7500, 0.0000],
               [0.0000, 0.0000, 0.2500, 0.2500, 0.0000, 0.0000]],

              [[0.2500, 0.2500, 0.5000, 0.5000, 0.0000, 0.0000],
               [0.2500, 0.5000, 1.0000, 1.0000, 0.7500, 0.5000],
               [0.2500, 0.5000, 0.5000, 0.7500, 1.2500, 0.7500],
               [0.2500, -0.5000, -0.5000, 0.7500, 0.7500, 0.2500],
               [0.0000, -0.7500, 0.0000, 1.0000, 0.2500, 0.0000],
               [0.0000, 0.0000, 0.5000, 0.5000, 0.0000, 0.0000]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_avg_pool2d_3(self):

        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        avg_layer = AveragePoolLayer(self.HE, kernel_size=(2, 2), stride=(1, 1), padding=(1, 0))

        result = cr.decrypt_matrix(self.HE, avg_layer(encrypted_image))

        expected_result = np.array(
            [[[[0.5000, 0.0000, -0.2500, 0.0000],
               [0.7500, 0.5000, 0.2500, 0.2500],
               [0.2500, 0.7500, 1.0000, 0.7500],
               [-0.5000, 0.0000, 1.0000, 0.7500],
               [-0.2500, 0.2500, 0.7500, 0.2500],
               [0.2500, 0.5000, 0.2500, 0.0000]],

              [[0.2500, 0.2500, 0.5000, 0.2500],
               [1.0000, 1.0000, 1.0000, 0.2500],
               [0.7500, 0.5000, 0.2500, 0.2500],
               [0.2500, 0.2500, 0.2500, 0.5000],
               [0.5000, 0.7500, 0.5000, 0.2500],
               [0.2500, 0.2500, 0.0000, 0.0000]]],

             [[[0.5000, 0.5000, 0.2500, 0.0000],
               [0.7500, 1.2500, 1.0000, 0.5000],
               [0.2500, 1.0000, 1.2500, 1.0000],
               [0.2500, 0.7500, 1.7500, 1.2500],
               [0.2500, 0.7500, 1.5000, 0.7500],
               [0.0000, 0.2500, 0.2500, 0.0000]],

              [[0.2500, 0.5000, 0.5000, 0.0000],
               [0.5000, 1.0000, 1.0000, 0.7500],
               [0.5000, 0.5000, 0.7500, 1.2500],
               [-0.5000, -0.5000, 0.7500, 0.7500],
               [-0.7500, 0.0000, 1.0000, 0.2500],
               [0.0000, 0.5000, 0.5000, 0.0000]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_avg_pool2d_4(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        avg_layer = AveragePoolLayer(self.HE, kernel_size=(2, 2), stride=(1, 2), padding=(1, 0))

        result = cr.decrypt_matrix(self.HE, avg_layer(encrypted_image))

        expected_result = np.array(
            [[[[0.5000, -0.2500],
               [0.7500, 0.2500],
               [0.2500, 1.0000],
               [-0.5000, 1.0000],
               [-0.2500, 0.7500],
               [0.2500, 0.2500]],

              [[0.2500, 0.5000],
               [1.0000, 1.0000],
               [0.7500, 0.2500],
               [0.2500, 0.2500],
               [0.5000, 0.5000],
               [0.2500, 0.0000]]],

             [[[0.5000, 0.2500],
               [0.7500, 1.0000],
               [0.2500, 1.2500],
               [0.2500, 1.7500],
               [0.2500, 1.5000],
               [0.0000, 0.2500]],

              [[0.2500, 0.5000],
               [0.5000, 1.0000],
               [0.5000, 0.7500],
               [-0.5000, 0.7500],
               [-0.7500, 1.0000],
               [0.0000, 0.5000]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_avg_pool2d_5(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        avg_layer = AveragePoolLayer(self.HE, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))

        result = cr.decrypt_matrix(self.HE, avg_layer(encrypted_image))

        expected_result = np.array(
            [[[[0.1667, -0.1667],
               [0.5000, 0.1667],
               [0.5000, 0.8333],
               [0.0000, 0.8333],
               [0.1667, 0.5000],
               [0.3333, 0.1667]],

              [[0.3333, 0.3333],
               [1.1667, 0.6667],
               [0.6667, 0.3333],
               [0.1667, 0.3333],
               [0.5000, 0.3333],
               [0.1667, 0.0000]]],

             [[[0.5000, 0.1667],
               [1.0000, 0.8333],
               [0.6667, 1.1667],
               [0.6667, 1.3333],
               [0.6667, 1.0000],
               [0.1667, 0.1667]],

              [[0.5000, 0.3333],
               [0.8333, 1.0000],
               [0.5000, 1.0000],
               [-0.1667, 0.6667],
               [0.0000, 0.6667],
               [0.3333, 0.3333]]]])

        self.assertTrue(np.allclose(result, expected_result, 0.01))

    def test__avg(self):
        """ Procedure:
                    1. Create a image in the form
                        [y, x]
                    2. Encrypt the image
                    3. Use __avg
                    ---------
                    Verification:
                    4. Verify the result is the expected
        """
        HE = Pyfhel()
        HE.contextGen(65532)
        HE.keyGen()
        HE.relinKeyGen(20, 30)
        # Shape of image is [1, 1, 5, 5]. Needed to use encrypt_matrix.
        image = np.array([[[
            [1, -1, 1, 0, 0]
            , [0, 1, -2, 1, 0]
            , [0, 0, 1, 1, 1]
            , [0, -3, 1, 1, 0]
            , [0, 1, 1, 0, 0]
        ]]])

        encrypted_image = cr.encrypt_matrix(HE, image)

        encrypted_result = avg._avg(HE, encrypted_image[0][0], kernel_size=(3, 3), stride=(2, 2))

        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = np.array(
            [[0.1111, 0.3333],
            [0.1111, 0.6667]])

        self.assertTrue(np.allclose(result, expected_result, 0.001))


class SquareLayerTests(unittest.TestCase):
    def test_square_layer(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(20, 100)

        image = np.array([
            [
                [
                    [1, 1, 1, 0, 0]
                    , [0, 1, 1, 1, 0]
                    , [0, 0, 1, 1, 1]
                    , [0, 0, 1, 1, 0]
                    , [0, 1, 1, 0, 0]
                ],
                [
                    [1, 0, 1, 1, 0]
                    , [0, 1, 2, 0, 0]
                    , [0, 0, 1, 0, 1]
                    , [0, 1, 1, 1, 0]
                    , [0, 1, 0, 0, 0]],
            ],
            [
                [
                    [1, 1, 1, 0, 0]
                    , [0, 1, 2, 1, 1]
                    , [0, 0, 1, 1, 1]
                    , [1, 0, 2, 1, 0]
                    , [0, 0, 1, 0, 0]
                ],
                [
                    [-1, 0, 2, 0, 0]
                    , [0, 1, 1, 1, 2]
                    , [1, 0, 0, 1, 1]
                    , [0, 0, 1, 1, 0]
                    , [0, 0, 2, 0, 0]
                ]
            ]
        ])

        expected_result = np.array([
            [
                [
                    [1, 1, 1, 0, 0]
                    , [0, 1, 1, 1, 0]
                    , [0, 0, 1, 1, 1]
                    , [0, 0, 1, 1, 0]
                    , [0, 1, 1, 0, 0]
                ],
                [
                    [1, 0, 1, 1, 0]
                    , [0, 1, 4, 0, 0]
                    , [0, 0, 1, 0, 1]
                    , [0, 1, 1, 1, 0]
                    , [0, 1, 0, 0, 0]],
            ],
            [
                [
                    [1, 1, 1, 0, 0]
                    , [0, 1, 4, 1, 1]
                    , [0, 0, 1, 1, 1]
                    , [1, 0, 4, 1, 0]
                    , [0, 0, 1, 0, 0]
                ],
                [
                    [1, 0, 4, 0, 0]
                    , [0, 1, 1, 1, 4]
                    , [1, 0, 0, 1, 1]
                    , [0, 0, 1, 1, 0]
                    , [0, 0, 4, 0, 0]
                ]
            ]
        ])

        encrypted_image = cr.encrypt_matrix(HE, image)
        encrypted_result = sq.square(HE, encrypted_image)

        result = cr.decrypt_matrix(HE, encrypted_result)

        self.assertTrue(np.allclose(expected_result, result))

    def test_square_layer2D(self):
        HE = Pyfhel()
        HE.contextGen(65)
        HE.keyGen()
        HE.relinKeyGen(20, 30)

        image = np.array([
                    [1, -2, 1, 3, 0]
                    , [0, 1, 4, 1, 0]
                    , [0, 2, 1, 1, 1]
                ])

        expected_result = np.array([
                    [1, 4, 1, 9, 0]
                    , [0, 1, 16, 1, 0]
                    , [0, 4, 1, 1, 1]
                ])

        encrypted_image = cr.encrypt_matrix(HE, image)
        encrypted_result = sq.square(HE, encrypted_image)

        result = cr.decrypt_matrix(HE, encrypted_result)

        self.assertTrue(np.allclose(expected_result, result))


class RencryptionLayerTests(unittest.TestCase):
    def test_rencryption_layer(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        image = np.array([
                      [1, 1, -1, 0, 0],
                      [1, 0, 1, 1, 0]])

        enc_image = cr.encrypt_matrix(HE, image)
        rencryption_layer = RencryptionLayer(HE)

        enc_result = rencryption_layer(enc_image)

        result = cr.decrypt_matrix(HE, enc_result)
        self.assertTrue(np.allclose(image, result))


class ReshapeLayerTests(unittest.TestCase):
    def test_reshape_layer(self):
        image = np.array([
            [
                [
                    [1, 1, -1, 0, 0]
                    , [0, 1, 1, 1, 0]
                    , [0, 0, 1, 1, 1]
                    , [0, -2, 1, 1, 0]
                    , [0, 1, 1, 0, 0]
                ],
                [
                    [1, 0, 1, 1, 0]
                    , [2, 1, 2, 0, 0]
                    , [0, 0, -1, 0, 1]
                    , [0, 1, 1, 1, 0]
                    , [0, 1, 0, 0, 0]],
            ],
            [
                [
                    [1, 1, 1, 0, 0]
                    , [0, 1, 2, 1, 1]
                    , [0, 0, 1, 1, 1]
                    , [1, 0, 2, 3, 0]
                    , [0, 0, 1, 0, 0]
                ],
                [
                    [1, 0, 2, 0, 0]
                    , [0, 1, 1, 1, 2]
                    , [1, 0, 0, 1, 1]
                    , [0, -3, 1, 1, 0]
                    , [0, 0, 2, 0, 0]
                ]
            ]
        ])

        reshape_layer = FlattenLayer()
        result = reshape_layer(image)

        self.assertEqual(result.shape, (2, 50))


if __name__ == '__main__':
    unittest.main()
