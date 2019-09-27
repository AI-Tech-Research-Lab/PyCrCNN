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

        new_weights = np.array(
            [
                [
                    [
                        [1, 0, 1]
                        , [0, 1, 0]
                        , [1, 0, 1]
                    ]
                ]
            ])
        bias = np.array([3])

        conv_layer = conv.ConvolutionalLayer(HE, new_weights
                                             , 1, 1, bias)

        image = np.array([[[
            [1, 1, 1, 0, 0]
            , [0, 1, 1, 1, 0]
            , [0, 0, 1, 1, 1]
            , [0, 0, 1, 1, 0]
            , [0, 1, 1, 0, 0]
        ]]])

        encrypted_image = cr.encrypt_matrix(HE, image)

        encrypted_result = conv_layer(encrypted_image)
        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = np.array([[[
            [4, 3, 4]
           ,[2, 4, 3]
           ,[2, 3, 4]
        ]]]
        )
        self.assertEqual(result.all(), (expected_result+3).all())

    def test_convolute(self):
        """ Procedure:
                    1. Create a image in the form
                        [n_images, n_layers, y, x]
                    2. Create a set of filters in the form
                        [n_filters, n_layers, y, x]
                    3. Encrypt the image
                    4. Encode the filters
                    5. Convolute
                    ---------
                    Verification:
                    6. Verify the result is the expected
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

        filters = np.array([
            [
                [
                    [1, 0, 1]
                  , [0, 1, 0]
                  , [1, 0, 1]
                ]
                ,
                [
                    [0, 1, 0]
                  , [1, 0, 1]
                  , [2, 0, 1]
                ]
            ]
            ,
            [
                [
                    [0, 1, 0]
                  , [1, 0, 1]
                  , [2, 0, 1]
                ]
                ,
                [
                    [1, 0, 1]
                  , [0, 1, 0]
                  , [1, 0, 1]
                ]
            ]
        ])

        encrypted_image = cr.encrypt_matrix(HE, image)
        encoded_filters = cr.encode_matrix(HE, filters)

        encrypted_result = conv.convolute(HE, encrypted_image, encoded_filters, 1, 1)

        result = cr.decrypt_matrix(HE, encrypted_result)

        expected_result = np.array([
            [[[ 7.,  5., 10.],
              [ 5.,  9.,  7.],
              [ 3.,  8.,  5.]],

             [[ 7.,  7.,  7.],
              [ 6.,  7.,  8.],
              [ 4.,  6.,  7.]]],

            [[[ 7.,  9.,  8.],
              [ 8.,  7., 10.],
              [ 5.,  4., 10.]],

             [[ 9.,  6., 10.],
              [ 8.,  7., 12.],
              [ 7.,  4.,  9.]]]])

        self.assertEqual(result.all(), expected_result.all())

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

        encrypted_result = conv.convolute2d(HE, encrypted_image[0][0], encoded_filter[0][0], 1, 1)

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
