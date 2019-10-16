import torch
import torchvision
import torchvision.transforms as transforms

from Pyfhel import Pyfhel

from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.crypto import crypto as cr
import numpy as np
import model

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


def get_max_error(HE, p_matrix, c_matrix):
    """Given a plain matrix and a cipher matrix, returns the max
    error between the two (the max delta) and the position in which
    this happens.

    Parameters
    ----------
    HE: Pyfhel
        Pyfhel object, needed to decrypt the cipher matrix
    p_matrix: np.array( dtype=float )
        Numpy array of plain values, can be either in the form
            [n_images, n_layers, y, x]
        of
            [n_images, value]
    c_matrix: np.array( dtype=PyCtxt )
        Numpy array of encrypted values, can be either in the form
            [n_images, n_layers, y, x]
        of
            [n_images, value]

    Returns
    -------
    max_error
    position: np.array(dtype = int)
        Position in the matrix in which the max error is found
    """
    try:
        dec_matrix = cr.decrypt_matrix(HE, c_matrix)
    except:
        dec_matrix = cr.decrypt_matrix_2d(HE, c_matrix)

    max_error = np.max(abs(p_matrix.detach().numpy() - dec_matrix))
    position = np.unravel_index(np.argmax(abs(p_matrix.detach().numpy() - dec_matrix)), p_matrix.shape)

    return max_error, position


def get_min_noise(HE, matrix):
    """Returns the minimum noise budget found in the entire cipher matrix

    Parameters
    ----------
    HE: Pyfhel
        Pyfhel obcjet
    matrix: np.array( dtype=PyCtxt )
        Encrypted matrix, either in the form
            [n_image, n_layer, y, x]
        of
            [n_images, values]

    Returns
    -------
    min_noise
    """
    min_noise = 10000000
    try:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                for z in range(0, len(matrix[0][0])):
                    for k in range(0, len(matrix[0][0][0])):
                        if HE.noiseLevel(matrix[i][j][z][k]) < min_noise:
                            min_noise = HE.noiseLevel(matrix[i][j][z][k])
                        if HE.noiseLevel(matrix[i][j][z][k]) < 1:
                            return 0
    except:
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[0])):
                if HE.noiseLevel(matrix[i][j]) < min_noise:
                    min_noise = HE.noiseLevel(matrix[i][j])
                if HE.noiseLevel(matrix[i][j]) < 1:
                    return 0
    return min_noise


def ask_encryption_parameters():
    """Asks for encryption parameters and returns an ordered list
    of them

    Returns
    -------
    parameters_list: List composed by tuples which contains the parameters
    given in input in the form
        (coefficient_modulus, plaintext_modulus, security_level, polynomial_base)
    """
    parameters_list = []

    use_case = 0
    get_another_case = True

    while get_another_case:
        print("---- CASE " + str(use_case) + " ----")
        coefficient_modulus = int(input("Enter a coefficient modulus (default=2048): ") or 2048)
        plaintext_modulus = int(input("Enter a plaintext modulus: "))
        security_level = int(input("Enter a security level (default=128): ") or 128)
        polynomial_base = int(input("Enter a polynomial base (default=2): ") or 2)
        parameters_list.append((coefficient_modulus, plaintext_modulus, security_level, polynomial_base))

        if str(input("Another one? [y/N]: ")) == "y":
            use_case = use_case + 1
        else:
            get_another_case = False

    return parameters_list


def test_net(HE, net, encoded_layers, images, verbose):
    enc_images = cr.encrypt_matrix(HE, images)
    net_iterator = net.children()

    if verbose:
        print("Min noise initial= ", get_min_noise(HE, enc_images))

    for layer in encoded_layers:
        enc_images = layer(enc_images)
        if not (type(layer) == RencryptionLayer):
            images = next(net_iterator)(images)

        if verbose and not (type(layer) == FlattenLayer) and not (type(layer) == RencryptionLayer):
            print("\n------------ INTERMEDIATE STATS ------------------------")
            max_error_value, max_error_position = get_max_error(HE, images, enc_images)
            print("Max error = " + str(max_error_value) + " at " + str(max_error_position))
            print("Plain value = ", images[max_error_position].detach().numpy(), " , cipher value = "
                  , HE.decryptFrac(enc_images[max_error_position]))
            print("Avg value= ", np.average(images.detach().numpy()))
            print("Min noise= ", get_min_noise(HE, enc_images))

    if verbose:
        print("\n------------ FINAL RESULTS --------------------------")
        print(images)
        print(cr.decrypt_matrix_2d(HE, enc_images))

    dec_matrix = cr.decrypt_matrix_2d(HE, enc_images)
    final_error = np.max(abs(images.detach().numpy() - dec_matrix))
    return final_error

# model.py should be in the root Python is runned and should contain the network definition
model.net.load_state_dict(torch.load("./model.pt"))
model.net.eval()
plain_net = model.net

test_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=False
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

test_loader = torch.utils.data.DataLoader(test_set
    ,batch_size=1
)
batch = next(iter(test_loader))
images, labels = batch

encryption_parameters = ask_encryption_parameters()
max_error_results = []

debug = str(input("Debug? y/N: "))
rencrypt_position = int(input("Where to put rencryption? (default=no rencryption): ") or -1)

print("-----------------------")

if debug == "y":
    debug = True
else:
    debug = False

for i in range(0, len(encryption_parameters)):
    print("\nCASE ", i, "###############")
    HE = Pyfhel()
    HE.contextGen(m=encryption_parameters[i][0],
                  p=encryption_parameters[i][1],
                  sec=encryption_parameters[i][2],
                  base=encryption_parameters[i][3])
    HE.keyGen()
    HE.relinKeyGen(20, 5)
    encoded_net = build_from_pytorch(HE, plain_net, rencrypt_position)
    max_error_results.append(test_net(HE, plain_net, encoded_net, images, debug))

print("-----------------------")
print("Tested values: ")
for i in range(0, len(encryption_parameters)):
    print(encryption_parameters[i])

print("-----------------------")

print("Obtained errors: ")
for i in range(0, len(max_error_results)):
    print(max_error_results[i])

print("-----------------------")
