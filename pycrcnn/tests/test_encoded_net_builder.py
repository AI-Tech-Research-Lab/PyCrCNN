import unittest
import torch.nn as nn
from Pyfhel.Pyfhel import Pyfhel
import numpy as np

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


def decode_matrix(HE, matrix):
    n_matrixes = len(matrix)
    n_layers = len(matrix[0])
    n_rows = len(matrix[0][0])
    n_columns = len(matrix[0][0][0])
    result = np.empty((n_matrixes, n_layers, n_rows, n_columns), dtype=float)

    for n_matrix in range(0, n_matrixes):
        for n_layer in range(0, n_layers):
            for i in range(0, n_rows):
                for k in range(0, n_columns):
                    result[n_matrix][n_layer][i][k] = HE.decodeFrac(
                        matrix[n_matrix][n_layer][i][k]
                    )
    return result

class NetBuilderTester(unittest.TestCase):

    def test_encoded_net_builder(self):

        plain_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 8, out_features=64),
        )

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()

        encoded_net = build_from_pytorch(HE, plain_net, 1)

        self.assertEqual(plain_net[0].weight.detach().numpy().all(),
                         decode_matrix(HE, encoded_net[0].weights).all())
