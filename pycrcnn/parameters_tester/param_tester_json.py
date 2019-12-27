import multiprocessing

import jsonpickle
import torch
import torchvision
import numpy as np
import torch.nn as nn
from Pyfhel import Pyfhel
from torchvision import transforms

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix
from pycrcnn.functional.flatten_layer import FlattenLayer
from pycrcnn.functional.rencryption_layer import RencryptionLayer
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch
from pycrcnn.parameters_tester.utils.utils import get_max_error, get_min_noise

jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)


def get_parameter(option):
    with open("./encryption_parameters.json", "r") as f:
        parameters = jsonpickle.decode(f.read())
    return parameters[option]


def test_net(plain_net, layers, enc_params, starting_images, verbose, return_dicti, ind):

    final_partial_results = []
    max_error_results = []

    for i in range(0, len(enc_params)):

        HE = Pyfhel()
        HE.contextGen(m=enc_params[i]["m"],
                      p=enc_params[i]["p"],
                      sec=enc_params[i]["sec"],
                      base=enc_params[i]["base"])
        HE.keyGen()
        HE.relinKeyGen(20, 5)

        images = starting_images
        encoded_net = build_from_pytorch(HE, plain_net[min(layers):max(layers)+1])
        enc_images = encrypt_matrix(HE, images.detach().numpy())
        partial_results = [{
            "layer_name" : "start",
            "min_noise" : float(get_min_noise(HE, enc_images)),
            "max_error" : 0,
            "max_error_position" : str((0, 0, 0, 0)),
            "cipher_value" : 0,
            "plain_value" : 0
        }]
        net_iterator = plain_net.children()

        for layer in encoded_net:
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

        final_partial_results.append(partial_results)
        max_error_results.append(final_error)

    to_print = []
    for i in range(0, len(enc_params)):
        max_error = {"final_max_error": max_error_results[i]}
        if verbose:
            to_print.append([enc_params[i], max_error, final_partial_results[i]])
        else:
            to_print.append([enc_params[i], max_error])
    return_dicti[ind] = to_print

def param_test():

    # model.pt should be in the directory Python is launched and should be produced by saving the PyTorch model
    # with
    # torch.save(net, path)

    encryption_parameters = get_parameter("encryption_parameters")
    debug = get_parameter("debug")
    layers = get_parameter("layers")
    dataset = get_parameter("dataset")
    model_path = get_parameter("model_path")
    max_threads = get_parameter("max_threads")

    test_set = getattr(torchvision.datasets, dataset)(
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

    plain_net = torch.load(model_path)
    plain_net.eval()

    if debug == "y":
        debug = True
    else:
        debug = False

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    distributions = []

    if len(encryption_parameters) % max_threads == 0:
        n_threads = max_threads
        subparams_dim = len(encryption_parameters) // n_threads
        for i in range(0, n_threads):
            distributions.append([i * subparams_dim, i * subparams_dim + subparams_dim])
    else:
        n_threads = min(max_threads, len(encryption_parameters))
        subparams_dim = len(encryption_parameters) // n_threads
        for i in range(0, n_threads):
            distributions.append([i * subparams_dim, i * subparams_dim + subparams_dim])
        for k in range(0, (len(encryption_parameters) % n_threads)):
            distributions[k][1] += 1
            distributions[k + 1::] = [[x + 1, y + 1] for x, y in distributions[k + 1::]]

    for i in range(0, n_threads):
        processes.append(multiprocessing.Process(target=test_net, args=(plain_net, layers,
                                                                        encryption_parameters[distributions[i][0]:distributions[i][1]],
                                                                        images, debug, return_dict, i)))
        processes[-1].start()

    for p in processes:
        p.join()

    result = return_dict[0]
    for i in range(1, n_threads):
        result = result + return_dict[i]

    with open("./results.json", "w") as f:
        f.write(jsonpickle.encode(result))


if __name__ == '__main__':
    param_test()