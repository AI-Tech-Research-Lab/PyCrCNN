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

        conv_layer = conv.ConvolutionalLayer(HE, new_weights, stride=(2, 2), padding=(2, 2))

        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][0][2]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][0]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][1]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][1][2]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer.stride[0], 2)
        self.assertEqual(conv_layer.stride[1], 2)
        self.assertEqual(conv_layer.bias, None)

        bias = np.array([3])

        conv_layer2 = conv.ConvolutionalLayer(HE, new_weights, stride=(2, 2), bias=bias)

        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][0][2]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][0]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][1]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][1][2]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][0]), 1)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][1]), 0)
        self.assertEqual(HE.decodeFrac(conv_layer2.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer2.stride[0], 2)
        self.assertEqual(conv_layer2.stride[1], 2)
        self.assertEqual(HE.decodeFrac(conv_layer2.bias[0]), 3)

    def test_call(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        filters = np.array(
            [[[[1., 0., 1.],
               [0., 1., 0.],
               [1., 0., 1.]],

              [[0., 1., 0.],
               [1., 0., 1.],
               [2., 0., 1.]]],

             [[[0., 1., 0.],
               [1., 0., 1.],
               [2., 0., 1.]],

              [[1., 0., 1.],
               [0., 1., 0.],
               [1., 0., 1.]]]])

        bias = np.array([3., 4.])

        conv_layer = conv.ConvolutionalLayer(HE, filters, stride=(1, 1), bias=bias, padding=(1, 1))

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

        expected_result = np.array(
            [[[[6., 9., 9., 9., 5.],
               [6., 10., 8., 13., 4.],
               [5., 8., 12., 10., 8.],
               [6., 6., 11., 8., 6.],
               [4., 6., 7., 5., 4.]],

              [[8., 9., 10., 10., 6.],
               [6., 11., 11., 11., 8.],
               [6., 10., 11., 12., 9.],
               [6., 8., 10., 11., 6.],
               [6., 7., 8., 7., 5.]]],

             [[[6., 10., 9., 12., 6.],
               [6., 10., 12., 11., 8.],
               [4., 11., 10., 13., 11.],
               [5., 8., 7., 13., 6.],
               [3., 8., 6., 8., 4.]],

              [[8., 9., 12., 13., 7.],
               [6., 13., 10., 14., 10.],
               [6., 12., 11., 16., 11.],
               [4., 11., 8., 13., 7.],
               [5., 6., 9., 7., 5.]]]])


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

        encrypted_result = conv.convolute2d(encrypted_image[0][0], encoded_filter[0][0], (1, 1))

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
