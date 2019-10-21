import torch
import torchvision
import torchvision.transforms as transforms

from Pyfhel import Pyfhel

from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.crypto import crypto as cr
import numpy as np

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch
from pycrcnn.parameters_tester.utils.utils import get_max_error, get_min_noise


def ask_encryption_parameters():
    """Asks for encryption parameters and returns an ordered list of them

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
    enc_images = cr.encrypt_matrix(HE, images.detach().numpy())
    net_iterator = net.children()

    def print_stats():
        max_error_value, max_error_position = get_max_error(HE, images.detach().numpy(), enc_images)
        print("\n------------ INTERMEDIATE STATS ------------------------")
        print("Max error = ", max_error_value, " at ",  max_error_position)
        print("Plain value = ", images[max_error_position].detach().numpy(), " , cipher value = "
              , HE.decryptFrac(enc_images[max_error_position]))
        print("Avg value= ", np.average(images.detach().numpy()))
        print("Min noise= ", get_min_noise(HE, enc_images))

    if verbose:
        print("Min noise initial= ", get_min_noise(HE, enc_images))

    for layer in encoded_layers:
        enc_images = layer(enc_images)
        if not (type(layer) == RencryptionLayer):
            images = next(net_iterator)(images)

        if verbose and not (type(layer) == FlattenLayer) and not (type(layer) == RencryptionLayer):
            print_stats()

    dec_matrix = cr.decrypt_matrix(HE, enc_images)
    final_error = np.max(abs(images.detach().numpy() - dec_matrix))
    if verbose:
        print("\n------------ FINAL RESULTS --------------------------")
        print(images)
        print(dec_matrix)

    return final_error


def param_test():

    # model.pt should be in the directory Python is launched and should be produced by saving the PyTorch model
    # with
    # torch.save(net, path)

    plain_net = torch.load("./model_full.pt")
    plain_net.eval()

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
    rencrypt_positions = [int(x) for x in input("Where to put rencryption? (default=no rencryption): ").split()]

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
        encoded_net = build_from_pytorch(HE, plain_net, rencrypt_positions)
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


if __name__ == '__main__':
    param_test()