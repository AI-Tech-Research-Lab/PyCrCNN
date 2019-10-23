import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.convolutional import convolutional_layer as conv
from pycrcnn.convolutional.convolutional_layer import ConvolutionalLayer
from pycrcnn.crypto import crypto as cr


class MyTestCase(unittest.TestCase):
    
    def setUp(self) -> None:
        self.HE = Pyfhel()
        self.HE.contextGen(65537)
        self.HE.keyGen()
        self.HE.relinKeyGen(20, 100)

        self.image = np.array(
         [[[[1., 1., 1., 0., 0.],
          [0., 1., 1., 1., 0.],
          [0., 0., 1., 1., 1.],
          [0., 0., 1., 1., 0.],
          [0., 1., 1., 0., 0.]],

         [[1., 0., 1., 1., 0.],
          [0., 1., 2., 0., 0.],
          [0., 0., 1., 0., 1.],
          [0., 1., 1., 1., 0.],
          [0., 1., 0., 0., 0.]]],


        [[[1., 1., 1., 0., 0.],
          [0., 1., 2., 1., 1.],
          [0., 0., 1., 1., 1.],
          [1., 0., 2., 1., 0.],
          [0., 0., 1., 0., 0.]],

         [[1., 0., 2., 0., 0.],
          [0., 1., 1., 1., 2.],
          [1., 0., 0., 1., 1.],
          [0., 0., 1., 1., 0.],
          [0., 0., 2., 0., 0.]]]])

    def test_init(self):
        
        new_weights = np.array(
            [[[[1, 0, 1],
               [0, 1, 0],
              [1, 0, 1]]]])

        conv_layer = ConvolutionalLayer(self.HE, new_weights, stride=(2, 2), padding=(2, 2))

        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][0][0]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][0][1]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][0][2]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][1][0]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][1][1]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][1][2]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][2][0]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][2][1]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer.stride[0], 2)
        self.assertEqual(conv_layer.stride[1], 2)
        self.assertEqual(conv_layer.bias, None)

        bias = np.array([3])

        conv_layer2 = ConvolutionalLayer(self.HE, new_weights, stride=(2, 2), bias=bias)

        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][0][0]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][0][1]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][0][2]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][1][0]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][1][1]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][1][2]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][2][0]), 1)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][2][1]), 0)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.weights[0][0][2][2]), 1)

        self.assertEqual(conv_layer2.stride[0], 2)
        self.assertEqual(conv_layer2.stride[1], 2)
        self.assertEqual(self.HE.decodeFrac(conv_layer2.bias[0]), 3)

    def test_conv_pool_1(self):
        
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
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
        
        bias = np.array([3, 4])

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 1), padding=(1, 1), bias=bias)

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

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

        self.assertTrue(np.allclose(result, expected_result))

    def test_conv_pool_2(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
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

        bias = np.array([3, 4])

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 1), padding=(0, 0), bias=bias)

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array(
            [[[[10., 8., 13.],
               [8., 12., 10.],
               [6., 11., 8.]],

              [[11., 11., 11.],
               [10., 11., 12.],
               [8., 10., 11.]]],

             [[[10., 12., 11.],
               [11., 10., 13.],
               [8., 7., 13.]],

              [[13., 10., 14.],
               [12., 11., 16.],
               [11., 8., 13.]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_conv_pool_3(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
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

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 1), padding=(1, 0))

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array(
            [[[[6., 6., 6.],
               [7., 5., 10.],
               [5., 9., 7.],
               [3., 8., 5.],
               [3., 4., 2.]],

              [[5., 6., 6.],
               [7., 7., 7.],
               [6., 7., 8.],
               [4., 6., 7.],
               [3., 4., 3.]]],

             [[[7., 6., 9.],
               [7., 9., 8.],
               [8., 7., 10.],
               [5., 4., 10.],
               [5., 3., 5.]],

              [[5., 8., 9.],
               [9., 6., 10.],
               [8., 7., 12.],
               [7., 4., 9.],
               [2., 5., 3.]]]])

        self.assertTrue(np.allclose(result, expected_result, 0.01))

    def test_conv_pool_4(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
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

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 2), padding=(1, 0))

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array(
            [[[[6., 6.],
               [7., 10.],
               [5., 7.],
               [3., 5.],
               [3., 2.]],

              [[5., 6.],
               [7., 7.],
               [6., 8.],
               [4., 7.],
               [3., 3.]]],

             [[[7., 9.],
               [7., 8.],
               [8., 10.],
               [5., 10.],
               [5., 5.]],

              [[5., 9.],
               [9., 10.],
               [8., 12.],
               [7., 9.],
               [2., 3.]]]])

        self.assertTrue(np.allclose(result, expected_result))

    def test_conv_pool_5(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
            [[[[-0.2205, 0.2531, 0.1530],
               [-0.1319, -0.0528, -0.1708]],

              [[0.1650, 0.2708, -0.2650],
               [-0.2814, -0.2790, -0.2479]]],

             [[[0.2485, 0.2866, 0.2389],
               [0.0338, -0.1652, 0.2492]],

              [[0.1184, -0.2400, 0.2176],
               [0.1023, 0.0383, 0.2524]]]])

        bias = np.array([0.1049, 0.1855])

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 1), padding=(0, 0), bias=bias)

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array(
            [[[[-0.8078, -1.0514, -0.4271],
               [-0.1668, 0.4946, -0.4172],
               [-0.7048, -0.2500, -0.5544],
               [-0.2388, 0.2159, 0.4415]],

              [[1.9226, 0.9948, 0.3855],
               [1.4077, 0.7201, 1.4299],
               [1.1818, 0.9479, 1.3047],
               [0.5241, 0.7778, 0.6328]]],

             [[[-0.9957, -0.5373, -1.3290],
               [0.2176, 0.2430, -0.9064],
               [-0.2985, -0.5572, -0.5805],
               [-0.7412, 0.1592, -0.3417]],

              [[2.1369, 0.5861, 1.4678],
               [1.2789, 1.6785, 1.9300],
               [1.3274, 1.1380, 0.9800],
               [1.8834, 0.8865, 1.0859]]]])

        self.assertTrue(np.allclose(result, expected_result, 0.01))

    def test_conv_pool_6(self):
        encrypted_image = cr.encrypt_matrix(self.HE, self.image)

        weights = np.array(
            [[[[-0.2400, -0.0442, 0.1996],
               [0.1662, 0.0612, 0.0044]],

              [[0.0612, -0.1516, -0.1771],
               [0.0139, 0.2774, 0.1465]]],

             [[[0.2544, 0.0460, 0.0820],
               [0.0862, 0.1762, 0.0321]],

              [[-0.1757, 0.2729, 0.2850],
               [0.2734, -0.1417, -0.1980]]]])

        bias = np.array([-0.1245, -0.0447])

        conv_layer = ConvolutionalLayer(self.HE, weights, stride=(1, 2), padding=(1, 0), bias=bias)

        result = cr.decrypt_matrix(self.HE, conv_layer(encrypted_image))

        expected_result = np.array(
            [[[[0.2676, 0.3329],
               [0.3110, -0.1998],
               [-0.3240, 0.1058],
               [0.3263, 0.1936],
               [0.0895, -0.3329],
               [-0.1207, -0.3645]],

              [[0.3252, 0.1732],
               [0.1177, 1.1160],
               [0.7604, 0.2742],
               [0.0148, 0.8410],
               [0.6619, 0.4391],
               [0.3562, 0.2097]]],

             [[[0.4141, 0.0694],
               [-0.0082, 0.7401],
               [-0.0799, -0.2380],
               [0.4577, 0.1470],
               [0.1550, -0.5452],
               [-0.2791, -0.2421]],

              [[0.1272, 0.5882],
               [0.6328, -0.0253],
               [1.0287, 1.2141],
               [-0.1859, 1.3758],
               [0.2949, 1.2402],
               [0.6073, -0.1417]]]])

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
