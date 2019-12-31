import multiprocessing
import tempfile
import time

import jsonpickle
import numpy as np
import torch
import torchvision
from torchvision import transforms

from Pyfhel import Pyfhel
from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch

from pycrcnn.crypto.crypto import encrypt_matrix, decrypt_matrix


def local_execution(data, parameters, debug=False):

    def compute(local_HE, local_data, local_encoded_net, local_return_dict=None, ind=None):
        encrypted_data = encrypt_matrix(local_HE, local_data)

        for layer in local_encoded_net:
            encrypted_data = layer(encrypted_data)

        local_result = decrypt_matrix(local_HE, encrypted_data)

        if local_return_dict is None:
            return local_result
        else:
            local_return_dict[ind] = local_result

    encryption_parameters = parameters["encryption_parameters"]
    max_threads = parameters["max_threads"]
    layers = parameters["layers"]

    HE = Pyfhel()
    HE.contextGen(m=encryption_parameters[0]["m"],
                  p=encryption_parameters[0]["p"],
                  sec=encryption_parameters[0]["sec"],
                  base=encryption_parameters[0]["base"])
    HE.keyGen()
    HE.relinKeyGen(20, 5)

    plain_net = torch.load("./model.pt")
    plain_net.eval()

    # Choose how many layers encode
    plain_net = plain_net[min(layers):max(layers) + 1]

    if debug:
        start_time = time.time()
        encoded_net = build_from_pytorch(HE, plain_net)
        encoding_time = time.time() - start_time
        print("Time for net encoding: ", encoding_time)
    else:
        encoded_net = build_from_pytorch(HE, plain_net)

    if max_threads == 1:
        result = compute(HE, data, encoded_net)
        return result

    else:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []

        distributions = []

        if len(data) % max_threads == 0:
            n_threads = max_threads
            subtensors_dim = len(data) // n_threads
            for i in range(0, n_threads):
                distributions.append([i * subtensors_dim, i * subtensors_dim + subtensors_dim])
        else:
            n_threads = min(max_threads, len(data))
            subtensors_dim = len(data) // n_threads
            for i in range(0, n_threads):
                distributions.append([i * subtensors_dim, i * subtensors_dim + subtensors_dim])
            for k in range(0, (len(data) % n_threads)):
                distributions[k][1] += 1
                distributions[k + 1::] = [[x + 1, y + 1] for x, y in distributions[k + 1::]]

        for i in range(0, n_threads):
            processes.append(
                multiprocessing.Process(target=compute, args=(HE, data[distributions[i][0]:distributions[i][1]]
                                                                       , encoded_net, return_dict, i)))
            processes[-1].start()

        for p in processes:
            p.join()

        data = np.array(return_dict[0])
        for i in range(1, n_threads):
            data = np.concatenate((data, return_dict[i]))

        return data


def test():
    with open("./parameters.json", "r") as f:
        params = jsonpickle.decode(f.read())

    try:
        dataset = params["dataset"]
        batch_size = params["batch_size"]

        test_set = getattr(torchvision.datasets, dataset)(
            root='./data'
            , train=False
            , download=True
            , transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
        batch = next(iter(test_loader))
        images, labels = batch
        images = images.detach().numpy()

    except:
        with open("./input_images.json", "rb") as f:
            images = jsonpickle.decode(f.read())

    start_time = time.time()
    results = local_execution(images, params, True)
    remote_time = time.time() - start_time

    # EXTRA DEBUG TO CHECK THE RESULTS
    print("DEBUG: Plain results...")

    start_time = time.time()

    plain_net = torch.load("./model.pt")
    plain_net.eval()

    layers = params["layers"]
    plain_net = plain_net[min(layers):max(layers) + 1]
    results_plain = plain_net(torch.tensor(images)).detach().numpy()

    torch_time = time.time() - start_time

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print(results.shape)
    print(results_plain.shape)

    print(results - results_plain)

    print("Time for remote execution: ", remote_time)
    print("Time for local execution: ", torch_time)


if __name__ == '__main__':
    test()
