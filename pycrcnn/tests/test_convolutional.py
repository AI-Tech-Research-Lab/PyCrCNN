import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.convolutional import convolutional_layer as conv
from pycrcnn.crypto import crypto as cr


class MyTestCase(unittest.TestCase):

    def test_init(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        new_weights = np.array(
            [
                [
                    [
                        [1, 0, 1]
                       ,[0, 1, 0]
                       ,[1, 0, 1]
                    ]
                ]
            ])

        conv_layer = conv.ConvolutionalLayer(HE, new_weights
                                             , 2, 2)

        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][2]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][0]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][1]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][2]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer.x_stride, 2)
        self.assertEqual(conv_layer.y_stride, 2)
        self.assertEqual(conv_layer.bias, None)

        bias = np.array([3])

        conv_layer2 = conv.ConvolutionalLayer(HE, new_weights
                                                       , 2, 2, bias)

        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][2]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][0]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][1]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][2]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer2.x_stride, 2)
        self.assertEqual(conv_layer2.y_stride, 2)
        self.assertEqual(HE.decodeFrac(conv_layer2.bias[0]), 3)

    def test_call(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        filters = np.array(
         [[[[ 0.1915, -0.0583, -0.0125],
          [-0.1236, -0.1610,  0.0203],
          [ 0.1431,  0.2236,  0.0757]],

         [[-0.2123, -0.1988, -0.1835],
          [ 0.2101, -0.1124, -0.1516],
          [ 0.1216, -0.2243, -0.0824]]],


        [[[-0.2148,  0.2113, -0.1670],
          [-0.0213,  0.0091, -0.0951],
          [-0.0953, -0.1923, -0.0630]],

         [[-0.1646, -0.0732, -0.1820],
          [-0.0294,  0.1245,  0.2197],
          [-0.2307,  0.0445, -0.1276]]]])

        bias = np.array([-0.0983,  0.1762])

        conv_layer = conv.ConvolutionalLayer(HE, filters
                                             , 1, 1, bias)

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

        encrypted_result = conv_layer(encrypted_image)
        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = np.array([
         [[[-0.9363, -0.5513,  0.2994],
          [-1.0971, -0.7263, -0.3314],
          [-0.4630, -0.0742, -0.4171]],

         [[-0.0535, -0.1809, -1.0565],
          [-0.2381, -0.8359, -0.5473],
          [-0.1343, -0.1421, -0.3532]]],


        [[[-0.7440, -0.6249, -0.7889],
          [-0.1213, -0.7064, -0.6852],
          [-0.6468, -1.1432, -0.2841]],

         [[-0.6534, -0.1401, -0.3956],
          [-0.6751, -0.5997, -1.1430],
          [-0.4654,  0.2028, -0.7447]]]])

        self.assertTrue(np.allclose(result, expected_result, 0.01))


    def test_convolute2d(self):
        """ Procedure:
                    1. Create a image in the form
                        [y, x]
                    2. Create a filter in the form
                        [y, x]
                    3. Encrypt the image
                    4. Encode the filter
                    5. Convolute
                    ---------
                    Verification:
                    6. Verify the result is the expected
        """
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)
        # Shape of image is [1, 1, 5, 5]. Needed to use encrypt_matrix.
        image = np.array([[[
            [1, 1, 1, 0, 0]
            , [0, 1, 1, 1, 0]
            , [0, 0, 1, 1, 1]
            , [0, 0, 1, 1, 0]
            , [0, 1, 1, 0, 0]
        ]]])

        # Shape of filter is [1, 1, 3, 3]. Needed to use encode_matrix.
        filter_matrix = np.array([[[
            [1, 0, 1]
            , [0, 1, 0]
            , [1, 0, 1]
        ]]])

        encrypted_image = cr.encrypt_matrix(HE, image)
        encoded_filter = cr.encode_matrix(HE, filter_matrix)

        encrypted_result = conv.convolute2d(encrypted_image[0][0], encoded_filter[0][0], 1, 1)

        self.assertEqual(HE.decryptFrac(encrypted_result[0][0]), 4)
        self.assertEqual(HE.decryptFrac(encrypted_result[0][1]), 3)
        self.assertEqual(HE.decryptFrac(encrypted_result[0][2]), 4)
        self.assertEqual(HE.decryptFrac(encrypted_result[1][0]), 2)
        self.assertEqual(HE.decryptFrac(encrypted_result[1][1]), 4)
        self.assertEqual(HE.decryptFrac(encrypted_result[1][2]), 3)
        self.assertEqual(HE.decryptFrac(encrypted_result[2][0]), 2)
        self.assertEqual(HE.decryptFrac(encrypted_result[2][1]), 3)
        self.assertEqual(HE.decryptFrac(encrypted_result[2][2]), 4)


if __name__ == '__main__':
    unittest.main()
