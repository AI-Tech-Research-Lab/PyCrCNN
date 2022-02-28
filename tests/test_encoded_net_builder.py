import pytest
import torch.nn as nn
from Pyfhel.Pyfhel import Pyfhel
import numpy as np

from pycrcnn.crypto.crypto import decode_matrix
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


class TestNetBuilderTester:

    def test_encoded_net_builder(self):

        plain_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 8, out_features=64),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5),
            nn.Conv2d(in_channels=1, out_channels=1, bias=False, kernel_size=5, stride=(2, 2))
        )

        HE = Pyfhel()
        HE.contextGen(65537)
        HE.keyGen()

        encoded_net = build_from_pytorch(HE, plain_net)

        assert np.allclose(plain_net[0].weight.detach().numpy(),
                        decode_matrix(HE, encoded_net[0].weights))

        assert np.allclose(plain_net[0].bias.detach().numpy(),
                                    decode_matrix(HE, encoded_net[0].bias))

        assert plain_net[0].stride[0] == encoded_net[0].stride[0]
        assert plain_net[0].stride[1] == encoded_net[0].stride[1]

        assert plain_net[1].kernel_size == encoded_net[1].kernel_size
        assert plain_net[1].stride == encoded_net[1].stride

        assert type(encoded_net[2]) == FlattenLayer

        assert np.allclose(plain_net[4].weight.detach().numpy(),
                                    decode_matrix(HE, encoded_net[4].weights))

        assert encoded_net[4].bias is None

        assert plain_net[4].stride[0] == encoded_net[4].stride[0]
        assert plain_net[4].stride[1] == encoded_net[4].stride[1]
