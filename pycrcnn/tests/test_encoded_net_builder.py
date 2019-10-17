import unittest
import torch.nn as nn
from Pyfhel.Pyfhel import Pyfhel
import numpy as np

from pycrcnn.crypto.crypto import decode_matrix, decode_matrix, decode_matrix
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


class NetBuilderTester(unittest.TestCase):

    def test_encoded_net_builder(self):

        plain_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 8, out_features=64),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5, stride=2)
        )

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()

        encoded_net = build_from_pytorch(HE, plain_net, [2])
        print(encoded_net)

        self.assertTrue(np.allclose(plain_net[0].weight.detach().numpy(),
                        decode_matrix(HE, encoded_net[0].weights)))

        self.assertTrue(np.allclose(plain_net[0].bias.detach().numpy(),
                                    decode_matrix(HE, encoded_net[0].bias)))

        self.assertEqual(plain_net[0].stride[0],
                         encoded_net[0].x_stride)
        self.assertEqual(plain_net[0].stride[1],
                         encoded_net[0].y_stride)

        self.assertEqual(plain_net[1].kernel_size, encoded_net[1].kernel_size)
        self.assertEqual(plain_net[1].stride, encoded_net[1].stride)

        self.assertEqual(type(encoded_net[2]), RencryptionLayer)
        self.assertEqual(type(encoded_net[3]), FlattenLayer)

        self.assertTrue(np.allclose(plain_net[3].weight.detach().numpy(),
                                    decode_matrix(HE, encoded_net[4].weights)))
        self.assertTrue(np.allclose(plain_net[3].bias.detach().numpy(),
                                    decode_matrix(HE, encoded_net[4].bias)))

        self.assertEqual(encoded_net[5].bias, None)

        self.assertEqual(plain_net[5].stride[0],
                         encoded_net[6].x_stride)
        self.assertEqual(plain_net[5].stride[1],
                         encoded_net[6].y_stride)


if __name__ == '__main__':
    unittest.main()