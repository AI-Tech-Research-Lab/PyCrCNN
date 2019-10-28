import jsonpickle
import torch
import torchvision
import numpy as np
from Pyfhel import Pyfhel
from torchvision import transforms

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch
from pycrcnn.parameters_tester.utils.utils import get_max_error, get_min_noise

jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)


def get_parameter(option):
    with open("./encryption_parameters.json", "r") as f:
        parameters = jsonpickle.decode(f.read())
    return parameters[option]


def test_net(HE, net, encoded_layers, images, verbose):
    enc_images = encrypt_matrix(HE, images.detach().numpy())
    partial_results = [{
        "layer_name" : "start",
        "min_noise" : float(get_min_noise(HE, enc_images)),
        "max_error" : 0,
        "max_error_position" : str((0, 0, 0, 0)),
        "cipher_value" : 0,
        "plain_value" : 0
    }]
    net_iterator = net.children()

    for layer in encoded_layers:
        enc_images = layer(enc_images)
        if not (type(layer) == RencryptionLayer):
            images = next(net_iterator)(images)

        if verbose:
            max_error, max_error_position = get_max_error(HE, images.detach().numpy(), enc_images)
            partial_results.append({
                "layer_name": str(type(layer)),
                "min_noise": float(get_min_noise(HE, enc_images)),
                "max_error": float(max_error),
                "max_error_position": str(max_error_position),
                "cipher_value": float(HE.decryptFrac(enc_images[max_error_position])),
                "plain_value": float(images[max_error_position])
        })

    dec_matrix = decrypt_matrix(HE, enc_images)
    final_error = float(np.max(abs(images.detach().numpy() - dec_matrix)))

    return partial_results, final_error

def param_test():

    # model.pt should be in the directory Python is launched and should be produced by saving the PyTorch model
    # with
    # torch.save(net, path)

    plain_net = torch.load("./model_full.pt")
    plain_net.eval()

    test_set = torchvision.datasets.MNIST(
        root='./data'
        , train=False
        , download=True
        , transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_loader = torch.utils.data.DataLoader(test_set,batch_size=1)
    batch = next(iter(test_loader))
    images, labels = batch

    encryption_parameters = get_parameter("encryption_parameters")
    debug = get_parameter("debug")

    final_partial_results = []
    max_error_results = []

    if debug == "y":
        debug = True
    else:
        debug = False

    for i in range(0, len(encryption_parameters)):

        HE = Pyfhel()
        HE.contextGen(m=encryption_parameters[i]["m"],
                      p=encryption_parameters[i]["p"],
                      sec=encryption_parameters[i]["sec"],
                      base=encryption_parameters[i]["base"])
        HE.keyGen()
        HE.relinKeyGen(20, 5)
        rencrypt_positions = encryption_parameters[i]["rencrypt_positions"]
        encoded_net = build_from_pytorch(HE, plain_net, rencrypt_positions)
        partial_results, final_error = test_net(HE, plain_net, encoded_net, images, debug)

        final_partial_results.append(partial_results)
        max_error_results.append(final_error)

    with open("./results.json", "w") as f:
        to_print = []
        for i in range(0, len(encryption_parameters)):
            max_error = {"final_max_error": max_error_results[i]}
            if debug:
                to_print.append([encryption_parameters[i], max_error, final_partial_results[i]])
            else:
                to_print.append([encryption_parameters[i], max_error])

        f.write(jsonpickle.encode(to_print))


if __name__ == '__main__':
    param_test()