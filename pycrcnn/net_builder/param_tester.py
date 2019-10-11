import warnings

import numpy as np
import torch.nn.functional as F
from Pyfhel import Pyfhel

from pycrcnn.convolutional.convolutional_layer import ConvolutionalLayer
from pycrcnn.crypto import crypto as cr
from pycrcnn.functional import average_pool as avg
from pycrcnn.functional import square_layer as sq
from pycrcnn.functional.average_pool import AveragePoolLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.functional.reshape_layer import ReshapeLayer
from pycrcnn.linear.linear_layer import LinearLayer
from pycrcnn.net_builder import pytorch_net as pytorch

warnings.filterwarnings("ignore")


def max_error(HE, p_matrix, c_matrix):
    try:
        dec_matrix = cr.decrypt_matrix(HE, c_matrix)
    except:
        dec_matrix = cr.decrypt_matrix_2d(HE, c_matrix)
    max_index = np.unravel_index(np.argmax(abs(p_matrix.detach().numpy() - dec_matrix)), p_matrix.shape)
    max_error = np.max(abs(p_matrix.detach().numpy() - dec_matrix))
    print(max_error, ", ", max_index)
    print("Plain value= ", p_matrix[max_index].detach().numpy(), ", cipher value= ", HE.decryptFrac(c_matrix[max_index]))
    return max_error


def min_noise(HE, matrix):
    min_noise = 10000000
    try:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                for z in range (0, len(matrix[0][0])):
                    for k in range(0, len(matrix[0][0][0])):
                        if (HE.noiseLevel(matrix[i][j][z][k])<min_noise):
                            min_noise = HE.noiseLevel(matrix[i][j][z][k])
                        if (HE.noiseLevel(matrix[i][j][z][k])<1):
                            print("NOISE ALERT < 1")
                            return 0
    except:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                if (HE.noiseLevel(matrix[i][j])<min_noise):
                    min_noise = HE.noiseLevel(matrix[i][j])
                if (HE.noiseLevel(matrix[i][j])<1):
                    print("NOISE ALERT < 1")
                    return 0
    return min_noise


def test_net(HE, net, images, verbose=True):

    enc_images = cr.encrypt_matrix(HE, my_images)

    conv1 = ConvolutionalLayer(HE, net.conv1.weight.detach().numpy(), 1, 1,
                                    net.conv1.bias.detach().numpy())
    avg1 = AveragePoolLayer(HE, 2, 2)

    conv2 = ConvolutionalLayer(HE, net.conv2.weight.detach().numpy(), 1, 1,
                                    net.conv2.bias.detach().numpy())
    avg2 = AveragePoolLayer(HE, 2, 2)

    reshape1 = ReshapeLayer(4*4*64)

    rencryption1 = RencryptionLayer(HE)

    fc1 = LinearLayer(HE, net.fc1.weight.detach().numpy(), net.fc1.bias.detach().numpy())

    fc2 = LinearLayer(HE, net.fc2.weight.detach().numpy(), net.fc2.bias.detach().numpy())

    layers = [
        ("conv1", conv1),
        ("avg1", avg1),
        ("conv2", conv2),
        ("avg2", avg2),
        ("reshape1", reshape1),
        ("rencryption1", rencryption1),
        ("fc1", fc1),
        ("fc2", fc2)
    ]

    net_iterator = net.named_children()

    if verbose:
        print("Min noise initial= ", min_noise(HE, enc_images))
        print("Size Initial= ", enc_images[0][0][0][0].size())

    for layer in layers:
        enc_images = layer[1](enc_images)
        if type(layer[1]) == avg.AveragePoolLayer:
            images = F.avg_pool2d(images, layer[1].kernel_size, layer[1].stride)
        elif type(layer[1]) == sq.SquareLayer:
            images = images.pow(2)
        elif type(layer[1]) == ReshapeLayer:
            images = images.reshape(-1, layer[1].length)
        elif type(layer[1]) == RencryptionLayer:
            pass
        else:
            images = next(net_iterator)[1](images)
        if verbose and not(type(layer[1]) == ReshapeLayer) and not (type(layer[1]) == RencryptionLayer):
            print("\n------------ STATS AFTER " + layer[0] + "------------------------")
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


net = pytorch.load_net()

data_loader = pytorch.new_batch(1)
my_batch = next(iter(data_loader))
my_images, labels = my_batch

values = []
results = []

value = int(input("Enter a number: "))
while (value != 0):
    values.append(value)
    value = int(input("Enter a number: "))

debug = str(input("Debug? y/N: "))

print("-----------------------")

if debug == "y":
    debug = True
else:
    debug = False

for i in range(0, len(values)):
    HE = Pyfhel()
    HE.contextGen(p=values[i], m=2048, base=2)
    HE.keyGen()
    HE.relinKeyGen(20, 5)
    results.append((values[i], test_net(HE, net, my_images, debug)))

print("-----------------------")
print("Tested values: ")
for i in range(0, len(results)):
    print(results[i][0])

print("-----------------------")

print("Obtained errors: ")
for i in range(0, len(results)):
    print(results[i][1])

print("-----------------------")
