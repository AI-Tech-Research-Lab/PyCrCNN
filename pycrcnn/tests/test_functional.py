import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.functional import average_pool as avg
from pycrcnn.functional import square_layer as sq
from pycrcnn.crypto import crypto as cr


class MyTestCase(unittest.TestCase):
    def test_avg_pool2d(self):
        """ Procedure:
                    1. Create a image in the form
                        [n_images, n_layers, y, x]
                    2. Encrypt the image
                    4. Execute avg_pool2d
                    ---------
                    Verification:
                    5. Verify the result is the expected
        """
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)
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
                    [1, 0, 2, 0, 0]
                    , [0, 1, 1, 1, 2]
                    , [1, 0, 0, 1, 1]
                    , [0, 0, 1, 1, 0]
                    , [0, 0, 2, 0, 0]
                ]
            ]
        ])

        encrypted_image = cr.encrypt_matrix(HE, image)

        encrypted_result = avg.avg_pool2d(HE, encrypted_image, 2, 1)

        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = np.array([
          [[[0.7500, 1.0000, 0.7500, 0.2500],
            [0.2500, 0.7500, 1.0000, 0.7500],
            [0.0000, 0.5000, 1.0000, 0.7500],
            [0.2500, 0.7500, 0.7500, 0.2500]],

           [[0.5000, 1.0000, 1.0000, 0.2500],
            [0.2500, 1.0000, 0.7500, 0.2500],
            [0.2500, 0.7500, 0.7500, 0.5000],
            [0.5000, 0.7500, 0.5000, 0.2500]]],

          [[[0.7500, 1.2500, 1.0000, 0.5000],
            [0.2500, 1.0000, 1.2500, 1.0000],
            [0.2500, 0.7500, 1.2500, 0.7500],
            [0.2500, 0.7500, 1.0000, 0.2500]],

           [[0.5000, 1.0000, 1.0000, 0.7500],
            [0.5000, 0.5000, 0.7500, 1.2500],
            [0.2500, 0.2500, 0.7500, 0.7500],
            [0.0000, 0.7500, 1.0000, 0.2500]]]])

        self.assertEqual(result.all(), expected_result.all())

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
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(20, 30)
        # Shape of image is [1, 1, 5, 5]. Needed to use encrypt_matrix.
        image = np.array([[[
            [1, 1, 1, 0, 0]
            , [0, 1, 1, 1, 0]
            , [0, 0, 1, 1, 1]
            , [0, 0, 1, 1, 0]
            , [0, 1, 1, 0, 0]
        ]]])

        encrypted_image = cr.encrypt_matrix(HE, image)

        encrypted_result = avg._avg(HE, encrypted_image[0][0], 3, 2)

        result = cr.decrypt_matrix_2x2(HE, encrypted_result)

        expected_result = np.array(
            [[[[0.6667, 0.6667],
               [0.4444, 0.6667]]]])

        self.assertEqual(result.all(), expected_result.all())

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

        self.assertEqual(expected_result.all(), result.all())

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

        encrypted_image = cr.encrypt_matrix_2x2(HE, image)
        encrypted_result = sq.square2d(HE, encrypted_image)

        result = cr.decrypt_matrix_2x2(HE, encrypted_result)

        self.assertEqual(expected_result.all(), result.all())


if __name__ == '__main__':
    unittest.main()
