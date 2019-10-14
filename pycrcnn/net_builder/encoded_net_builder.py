import numpy as np

from pycrcnn.convolutional.convolutional_layer import ConvolutionalLayer
from pycrcnn.crypto import crypto as cr
from pycrcnn.functional.average_pool import AveragePoolLayer
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.linear.linear_layer import LinearLayer


def max_error(HE, p_matrix, c_matrix):
    try:
        dec_matrix = cr.decrypt_matrix(HE, c_matrix)
    except:
        dec_matrix = cr.decrypt_matrix_2d(HE, c_matrix)
    max_index = np.unravel_index(np.argmax(abs(p_matrix.detach().numpy() - dec_matrix)), p_matrix.shape)
    max_error = np.max(abs(p_matrix.detach().numpy() - dec_matrix))
    print(max_error, ", ", max_index)
    print("Plain value= ", p_matrix[max_index].detach().numpy(), ", cipher value= ",
          HE.decryptFrac(c_matrix[max_index]))
    return max_error


def min_noise(HE, matrix):
    min_noise = 10000000
    try:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                for z in range(0, len(matrix[0][0])):
                    for k in range(0, len(matrix[0][0][0])):
                        if HE.noiseLevel(matrix[i][j][z][k]) < min_noise:
                            min_noise = HE.noiseLevel(matrix[i][j][z][k])
                        if HE.noiseLevel(matrix[i][j][z][k]) < 1:
                            print("NOISE ALERT < 1")
                            return 0
    except:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                if HE.noiseLevel(matrix[i][j]) < min_noise:
                    min_noise = HE.noiseLevel(matrix[i][j])
                if HE.noiseLevel(matrix[i][j]) < 1:
                    print("NOISE ALERT < 1")
                    return 0
    return min_noise


def test_net(HE, net, images, rencrypt_position, verbose=True):
    # define the function blocks
    def conv_layer(layer):
        return ConvolutionalLayer(HE, layer.weight.detach().numpy(),
                                  layer.stride[0],
                                  layer.stride[1],
                                  layer.bias.detach().numpy())

    def lin_layer(layer):
        return LinearLayer(HE, layer.weight.detach().numpy(),
                           layer.bias.detach().numpy())

    def avg_pool_layer(layer):
        return AveragePoolLayer(HE, layer.kernel_size, layer.stride)

    def flatten_layer(layer):
        return FlattenLayer()

    # map the inputs to the function blocks
    options = {"Conv": conv_layer,
               "Line": lin_layer,
               "Flat": flatten_layer,
               "AvgP": avg_pool_layer
               }

    encoded_layers = []

    for index in range(0, len(net)):
        encoded_layers.append(options[str(net[index])[0:4]](net[index]))
        if rencrypt_position == index:
            encoded_layers.append(RencryptionLayer())

    enc_images = cr.encrypt_matrix(HE, images)
    net_iterator = net.children()

    if verbose:
        print("Min noise initial= ", min_noise(HE, enc_images))
        print("Size Initial= ", enc_images[0][0][0][0].size())

    for layer in encoded_layers:
        enc_images = layer(enc_images)
        if not (type(layer) == RencryptionLayer):
            images = next(net_iterator)(images)

        if verbose and not (type(layer) == FlattenLayer) and not (type(layer) == RencryptionLayer):
            print("\n------------ INTERMEDIATE STATS ------------------------")
            print("Max error = ", end='')
            max_error(HE, images, enc_images)
            print("Avg value= ", np.average(images.detach().numpy()))
            print("Min noise= ", min_noise(HE, enc_images))

    if verbose:
        print("\n------------ FINAL RESULTS --------------------------")
        print(images)
        print(cr.decrypt_matrix_2d(HE, enc_images))

    dec_matrix = cr.decrypt_matrix_2d(HE, enc_images)
    final_error = np.max(abs(images.detach().numpy() - dec_matrix))
    return final_error
