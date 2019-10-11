import unittest

import numpy as np
from Pyfhel import Pyfhel

from pycrcnn.linear import linear_layer as lin
from pycrcnn.crypto import crypto as cr


class MyTestCase(unittest.TestCase):
    def test_init(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        new_weights = np.array([
            [1, 2, 3, 4]
           ,[2, 3, 4, 5]
           ,[3, 4, 5, 6]
        ])

        linear_layer = lin.LinearLayer(HE, new_weights)

        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][0]), 1)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][1]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][2]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][3]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][0]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][1]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][2]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][3]), 5)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][0]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][1]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][2]), 5)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][3]), 6)

        self.assertEqual(linear_layer.bias, None)

        bias = np.array([2, 2, 2])

        linear_layer2 = lin.LinearLayer(HE, new_weights, bias)

        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][0]), 1)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][1]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][2]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[0][3]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][0]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][1]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][2]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[1][3]), 5)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][0]), 3)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][1]), 4)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][2]), 5)
        self.assertEqual(HE.decodeFrac(linear_layer.weights[2][3]), 6)

        self.assertEqual(HE.decodeFrac(linear_layer2.bias[0]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer2.bias[1]), 2)
        self.assertEqual(HE.decodeFrac(linear_layer2.bias[2]), 2)

    def test_call(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        new_weights = np.array([
            [1, 2, 3, 4]
           ,[2, 3, 4, 5]
           ,[3, 4, 5, 6]
        ])
        new_bias = np.array([1, 2, 3])

        linear_layer = lin.LinearLayer(HE, new_weights, new_bias)

        flattened_input = np.array([
            [1, 2, 3, 4]
            , [5, 6, 7, 8]
        ])
        encrypted_input = cr.encrypt_matrix_2d(HE, flattened_input)

        encrypted_result = linear_layer(encrypted_input)
        result = cr.decrypt_matrix_2d(HE, encrypted_result)

        expected_result = np.array([
            [31, 42, 53]
           ,[43, 83, 113]
        ])
        self.assertEqual(result.all(), expected_result.all())

    def test_linear_multiply(self):
        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()
        HE.relinKeyGen(30, 100)

        flattened_input = np.array([
                [1, 2, 3, 4]
               ,[5, 6, 7, 8]
        ])
        encrypted_input = cr.encrypt_matrix_2d(HE, flattened_input)

        weights = np.array([
            [1, 2, 3, 4]
           ,[2, 3, 4, 5]
           ,[3, 4, 5, 6]
        ])
        encoded_weights = cr.encode_matrix_2d(HE, weights)

        encrypted_result = lin.linear_multiply(HE, encrypted_input, encoded_weights)
        result = cr.decrypt_matrix_2d(HE, encrypted_result)

        expected_result = np.array([
            [30, 40, 50]
           ,[42, 81, 110]
        ])

        self.assertEqual(result.all(), expected_result.all())


if __name__ == '__main__':
    unittest.main()
