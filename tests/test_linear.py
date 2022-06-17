import pytest
import numpy as np

from pycrcnn.he.HE import MockHE
from pycrcnn.linear import linear_layer as lin


class TestsLinearLayer:
    @pytest.fixture
    def HE(self):
        HE = MockHE()
        return HE

    def test_init(self, HE):
        new_weights = np.array([
            [1, 2, 3, 4]
           ,[2, 3, 4, 5]
           ,[3, 4, 5, 6]
        ])

        linear_layer = lin.LinearLayer(HE, new_weights)

        assert HE.decode(linear_layer.weights[0][0]) == 1
        assert HE.decode(linear_layer.weights[0][1]) == 2
        assert HE.decode(linear_layer.weights[0][2]) == 3
        assert HE.decode(linear_layer.weights[0][3]) == 4
        assert HE.decode(linear_layer.weights[1][0]) == 2
        assert HE.decode(linear_layer.weights[1][1]) == 3
        assert HE.decode(linear_layer.weights[1][2]) == 4
        assert HE.decode(linear_layer.weights[1][3]) == 5
        assert HE.decode(linear_layer.weights[2][0]) == 3
        assert HE.decode(linear_layer.weights[2][1]) == 4
        assert HE.decode(linear_layer.weights[2][2]) == 5
        assert HE.decode(linear_layer.weights[2][3]) == 6

        assert linear_layer.bias == None

        bias = np.array([2, 2, 2])

        linear_layer2 = lin.LinearLayer(HE, new_weights, bias)

        assert HE.decode(linear_layer.weights[0][0]) == 1
        assert HE.decode(linear_layer.weights[0][1]) == 2
        assert HE.decode(linear_layer.weights[0][2]) == 3
        assert HE.decode(linear_layer.weights[0][3]) == 4
        assert HE.decode(linear_layer.weights[1][0]) == 2
        assert HE.decode(linear_layer.weights[1][1]) == 3
        assert HE.decode(linear_layer.weights[1][2]) == 4
        assert HE.decode(linear_layer.weights[1][3]) == 5
        assert HE.decode(linear_layer.weights[2][0]) == 3
        assert HE.decode(linear_layer.weights[2][1]) == 4
        assert HE.decode(linear_layer.weights[2][2]) == 5
        assert HE.decode(linear_layer.weights[2][3]) == 6

        assert HE.decode(linear_layer2.bias[0]) == 2
        assert HE.decode(linear_layer2.bias[1]) == 2
        assert HE.decode(linear_layer2.bias[2]) == 2

    def test_call(self, HE):
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
        encrypted_input = HE.encrypt_matrix(flattened_input)

        encrypted_result = linear_layer(encrypted_input)
        result = HE.decrypt_matrix(encrypted_result)

        expected_result = np.array([
            [31, 42, 53]
            ,[71, 98, 125]
        ])
        assert np.allclose(result, expected_result)
