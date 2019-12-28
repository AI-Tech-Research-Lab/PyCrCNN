import base64
import datetime
import multiprocessing
import os
import tempfile
import time

import boto3 as boto3
import jsonpickle
import numpy as np

import requests
import torch
import torch.nn as nn
from Pyfhel import Pyfhel
import jsonpickle
import json

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        return torch.pow(t, 2)

def main():
    """

    This function executes a POST request given files:
        - parameters: JSON file
        - input_image: JSON file

    Parameters contains both the encryption parameters requested, as p, m, sec level and base and the model requested,
    with the layers on which the computation is requested (for example: MNIST, layers from 0 to 3)

    Due to limitations in the serialization of Pyfhel objects, the following algorithm is executed:

        - Decode the input image from the JSON file in a numpy array
        - Creates the Pyfhel object, generate the keys
        - Encrypt the image
        - Create an array with all the b64 value of every ciphertext's saved file
        - Append these data to the parameters
        - Send the post request with the JSON containing parameters and data
        - After receiving the response, do the opposite of pass 4, i.e de-serialize the answer JSON in a
          numpy matrix of ciphertexts
        - After the result is reconstructed in a suitable numpy array of ciphertext, decrypt it to obtain the plain
          result
        - OPTIONAL: print both the result and the plain result, obtained doing the same computation locally to ensure
          the result is correct (for debug purposes)
    """

    # jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)

    with open("./input_image_dim1.json", "rb") as f:
        input_image = jsonpickle.decode(f.read())

    with open("./parameters.json", "r") as f:
        parameters = jsonpickle.decode(f.read())

    result = remote_execution_aws(input_image, parameters)

    # EXTRA DEBUG TO CHECK THE RESULTS
    print("DEBUG: Plain results...")
    plain_net = torch.load("./mnist.pt")
    plain_net.eval()

    plain_net = plain_net[0:4]
    results_plain = plain_net(torch.tensor(input_image)).detach().numpy()

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print(result.shape)
    print(results_plain.shape)

    print(result - results_plain)


def remote_execution_aws(data, parameters):

    def encode_ciphertext(c, temp_file):
        with (open(temp_file.name, "w+b")) as f:
            c.save(temp_file.name)
            bc = f.read()
            b64 = str(base64.b64encode(bc))[2:-1]
            return b64

    def decode_ciphertext(b64, temp_file):
        with (open(temp_file.name, "w+b")) as f:
            x = bytes(b64, encoding='utf-8')
            x = base64.decodebytes(x)
            f.write(x)
            c = HE.encryptFrac(0)
            c.load(temp_file.name, "float")
            return c

    def crypt_and_encode(HE, t, ret_dict, ind):
        temp_file = tempfile.NamedTemporaryFile()
        enc_image = encrypt_matrix(HE, t)

        data = [[[[encode_ciphertext(value, temp_file) for value in row]
                  for row in column]
                 for column in layer]
                for layer in enc_image]
        ret_dict[ind] = data

    def decode_and_decrypt(HE, t, ret_dict, ind):
        temp_file = tempfile.NamedTemporaryFile()
        enc_res = np.array([[[[decode_ciphertext(value, temp_file) for value in row]
                                 for row in column]
                                for column in layer]
                               for layer in t])
        dec_res = decrypt_matrix(HE, enc_res)
        ret_dict[ind] = dec_res

    def uploadS3(msg_path, bucket, ak, sk):
        s3c = boto3.client(
            's3',
            aws_access_key_id=ak,
            aws_secret_access_key=sk
        )

        if '/' in msg_path:
            pos = len(msg_path) - 1
            c = msg_path[pos]
            while (c != '/'):
                pos -= 1
                c = msg_path[pos]
            pos += 1
        ts = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
        try:
            in_file_name = 'input_payload_{}'.format(ts)
            s3c.upload_file(msg_path, bucket, 'input/{}'.format(in_file_name))
            return in_file_name
        except:
            raise Exception("error while uploading {}".format(msg_path))

    address = parameters["address"]
    ak = parameters["ak"]
    sk = parameters["sk"]
    bk = parameters["bk"]

    encryption_parameters = parameters["encryption_parameters"]

    max_threads = parameters["max_threads"]

    temp_file_payload = tempfile.NamedTemporaryFile()

    HE = Pyfhel()
    HE.contextGen(m=encryption_parameters[0]["m"],
                  p=encryption_parameters[0]["p"],
                  sec=encryption_parameters[0]["sec"],
                  base=encryption_parameters[0]["base"])
    HE.keyGen()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    distributions = []

    if len(data) % max_threads == 0:
        n_threads = max_threads
        subtensors_dim = len(data) // n_threads
        for i in range(0, n_threads):
            distributions.append([i*subtensors_dim, i*subtensors_dim + subtensors_dim])
    else:
        n_threads = min(max_threads, len(data))
        subtensors_dim = len(data) // n_threads
        for i in range(0, n_threads):
            distributions.append([i*subtensors_dim, i*subtensors_dim+subtensors_dim])
        for k in range(0, (len(data) % n_threads)):
            distributions[k][1] += 1
            distributions[k+1::] = [ [x+1, y+1] for x, y in distributions[k+1::]]

    for i in range(0, n_threads):
        processes.append(multiprocessing.Process(target=crypt_and_encode, args=(HE, data[distributions[i][0]:distributions[i][1]]
                                                                       , return_dict, i)))
        processes[-1].start()

    for p in processes:
        p.join()

    data = np.array(return_dict[0])
    for i in range(1, n_threads):
        data = np.concatenate((data, return_dict[i]))

    payload = parameters
    payload["data"] = data.tolist()

    with open(temp_file_payload.name, 'w') as fp:
        json.dump(payload, fp)

    remote_filename = uploadS3(temp_file_payload.name, bk, ak, sk)

    msg = {
        "access_key": ak,
        "secret_key": sk,
        "bucket": bk,
        "filename": remote_filename
    }
    jmsg = json.dumps(msg)

    r = requests.post(address, jmsg)

    if r.status_code == 200:
        print("Done.")
        out_file = r.json()

    print("Downloading response...")

    s3r = boto3.resource(
        's3',
        aws_access_key_id=ak,
        aws_secret_access_key=sk
    )

    obj = s3r.Object(bk, 'output/{}'.format(out_file))
    resp_payload = json.load(obj.get()['Body'])

    print("Cleaning S3 support files...")
    inp_tbd = s3r.Object(bk, "input/{}".format(remote_filename))
    out_tbd = s3r.Object(bk, "output/{}".format(out_file))
    inp_tbd.delete();
    out_tbd.delete();
    print("Done.")

    enc_result = resp_payload["data"]

    return_dict = manager.dict()
    processes = []
    for i in range(0, n_threads):
        processes.append(multiprocessing.Process(target=decode_and_decrypt, args=(HE, enc_result[distributions[i][0]:distributions[i][1]]
                                                                       , return_dict, i)))
        processes[-1].start()

    for p in processes:
        p.join()

    result = np.array(return_dict[0])
    for i in range(1, n_threads):
        result = np.concatenate((result, return_dict[i]))

    return result


def remote_execution_rest(data, parameters):

    encryption_parameters = parameters["encryption_parameters"]
    address = parameters["address"]
    max_threads = parameters["max_threads"]

    def encode_ciphertext(c, temp_file):
        with (open(temp_file.name, "w+b")) as f:
            c.save(temp_file.name)
            bc = f.read()
            b64 = str(base64.b64encode(bc))[2:-1]
            return b64

    def decode_ciphertext(b64, temp_file):
        with (open(temp_file.name, "w+b")) as f:
            x = bytes(b64, encoding='utf-8')
            x = base64.decodebytes(x)
            f.write(x)
            c = HE.encryptFrac(0)
            c.load(temp_file.name, "float")
            return c

    def crypt_and_encode(HE, t, ret_dict=None, ind=None):
        temp_file = tempfile.NamedTemporaryFile()
        enc_image = encrypt_matrix(HE, t)

        data = [[[[encode_ciphertext(value, temp_file) for value in row]
                  for row in column]
                 for column in layer]
                for layer in enc_image]
        data = np.array(data)

        if ret_dict is not None:
            ret_dict[ind] = data
        else:
            return data

    def decode_and_decrypt(HE, t, ret_dict=None, ind=None):
        temp_file = tempfile.NamedTemporaryFile()
        enc_res = np.array([[[[decode_ciphertext(value, temp_file) for value in row]
                                 for row in column]
                                for column in layer]
                               for layer in t])
        dec_res = decrypt_matrix(HE, enc_res)
        if ret_dict is not None:
            ret_dict[ind] = dec_res
        else:
            return dec_res


    HE = Pyfhel()
    HE.contextGen(m=encryption_parameters[0]["m"],
                  p=encryption_parameters[0]["p"],
                  sec=encryption_parameters[0]["sec"],
                  base=encryption_parameters[0]["base"])
    HE.keyGen()

    if max_threads == 1:
        data = crypt_and_encode(HE, data)

        payload = parameters
        payload["data"] = data.tolist()

        json_payload = jsonpickle.encode(payload)
        res = requests.post(address, json=json_payload)

        enc_result = jsonpickle.decode(res.content)["data"]

        result = decode_and_decrypt(HE, enc_result)

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
                distributions.append([i*subtensors_dim, i*subtensors_dim + subtensors_dim])
        else:
            n_threads = min(max_threads, len(data))
            subtensors_dim = len(data) // n_threads
            for i in range(0, n_threads):
                distributions.append([i*subtensors_dim, i*subtensors_dim+subtensors_dim])
            for k in range(0, (len(data) % n_threads)):
                distributions[k][1] += 1
                distributions[k+1::] = [ [x+1, y+1] for x, y in distributions[k+1::]]

        for i in range(0, n_threads):
            processes.append(multiprocessing.Process(target=crypt_and_encode, args=(HE, data[distributions[i][0]:distributions[i][1]]
                                                                           , return_dict, i)))
            processes[-1].start()

        for p in processes:
            p.join()

        data = np.array(return_dict[0])
        for i in range(1, n_threads):
            data = np.concatenate((data, return_dict[i]))

        payload = parameters
        payload["data"] = data.tolist()

        json_payload = jsonpickle.encode(payload)
        res = requests.post(address, json=json_payload)

        enc_result = jsonpickle.decode(res.content)["data"]

        return_dict = manager.dict()
        processes = []
        for i in range(0, n_threads):
            processes.append(multiprocessing.Process(target=decode_and_decrypt, args=(HE, enc_result[distributions[i][0]:distributions[i][1]]
                                                                           , return_dict, i)))
            processes[-1].start()

        for p in processes:
            p.join()

        result = np.array(return_dict[0])
        for i in range(1, n_threads):
            result = np.concatenate((result, return_dict[i]))

        return result


# def remote_execution_rest_singlethreading(data, parameters):
#
#     encryption_parameters = parameters["encryption_parameters"]
#     address = parameters["address"]
#
#     def encode_ciphertext(c, temp_file):
#         with (open(temp_file.name, "w+b")) as f:
#             c.save(temp_file.name)
#             bc = f.read()
#             b64 = str(base64.b64encode(bc))[2:-1]
#             return b64
#
#     def decode_ciphertext(b64, temp_file):
#         with (open(temp_file.name, "w+b")) as f:
#             x = bytes(b64, encoding='utf-8')
#             x = base64.decodebytes(x)
#             f.write(x)
#             c = HE.encryptFrac(0)
#             c.load(temp_file.name, "float")
#             return c
#
#     def crypt_and_encode(HE, t):
#         temp_file = tempfile.NamedTemporaryFile()
#         enc_image = encrypt_matrix(HE, t)
#
#         data = [[[[encode_ciphertext(value, temp_file) for value in row]
#                   for row in column]
#                  for column in layer]
#                 for layer in enc_image]
#         data = np.array(data)
#         return data
#
#     def decode_and_decrypt(HE, t):
#         temp_file = tempfile.NamedTemporaryFile()
#         enc_res = np.array([[[[decode_ciphertext(value, temp_file) for value in row]
#                                  for row in column]
#                                 for column in layer]
#                                for layer in t])
#         dec_res = decrypt_matrix(HE, enc_res)
#         return dec_res
#
#     HE = Pyfhel()
#     HE.contextGen(m=encryption_parameters[0]["m"],
#                   p=encryption_parameters[0]["p"],
#                   sec=encryption_parameters[0]["sec"],
#                   base=encryption_parameters[0]["base"])
#     HE.keyGen()
#
#     data = crypt_and_encode(HE, data)
#
#     payload = parameters
#     payload["data"] = data.tolist()
#
#     json_payload = jsonpickle.encode(payload)
#     res = requests.post(address, json=json_payload)
#
#     enc_result = jsonpickle.decode(res.content)["data"]
#
#     result = decode_and_decrypt(HE, enc_result)
#
#     return result


def test_rest():

    # jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)

    with open("./input_image_dim1.json", "rb") as f:
        input_image = jsonpickle.decode(f.read())

    with open("./parameters.json", "r") as f:
        parameters = jsonpickle.decode(f.read())

    start_time = time.time()
    result = remote_execution_rest(input_image, parameters)
    remote_time = time.time() - start_time

    # EXTRA DEBUG TO CHECK THE RESULTS
    print("DEBUG: Plain results...")

    start_time = time.time()

    plain_net = torch.load("./model.pt")
    plain_net.eval()

    plain_net = plain_net[0:4]
    results_plain = plain_net(torch.tensor(input_image)).detach().numpy()

    torch_time = time.time() - start_time

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    print(result.shape)
    print(results_plain.shape)

    print(result - results_plain)

    print("Time for remote execution: ", remote_time)
    print("Time for local execution: ", torch_time)


if __name__ == '__main__':
    test_rest()
