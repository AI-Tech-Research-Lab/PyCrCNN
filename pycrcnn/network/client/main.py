import base64
import tempfile

import jsonpickle
import numpy as np

import requests
import torch
from Pyfhel import Pyfhel

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix


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

    jsonpickle.set_encoder_options('json', indent=4, sort_keys=False)
    temp_file = tempfile.NamedTemporaryFile()

    def encode_ciphertext(c):
        with (open(temp_file.name, "w+b")) as f:
            c.save(temp_file.name)
            bc = f.read()
            b64 = str(base64.b64encode(bc))[2:-1]
            return b64

    def decode_ciphertext(b64):
        with (open(temp_file.name, "w+b")) as f:
            x = bytes(b64, encoding='utf-8')
            x = base64.decodebytes(x)
            f.write(x)
            c = HE.encryptFrac(0)
            c.load(temp_file.name, "float")
            return c

    with open("./input_image.json", "rb") as f:
        input_image = jsonpickle.decode(f.read())

    with open("./parameters.json", "r") as f:
        parameters = jsonpickle.decode(f.read())
        encryption_parameters = parameters["encryption_parameters"]
        address = parameters["address"]

    HE = Pyfhel()
    HE.contextGen(m=encryption_parameters[0]["m"],
                  p=encryption_parameters[0]["p"],
                  sec=encryption_parameters[0]["sec"],
                  base=encryption_parameters[0]["base"])
    HE.keyGen()

    print("DEBUG: Encrypting matrix...")
    enc_image = encrypt_matrix(HE, input_image)

    print("DEBUG: Encode for JSON...")
    data = [[[[encode_ciphertext(value) for value in row]
              for row in column]
             for column in layer]
            for layer in enc_image]

    print("DEBUG: Sending request...")
    payload = parameters
    payload["data"] = data

    res = requests.post(address, json=jsonpickle.encode(payload))
    enc_result = jsonpickle.decode(res.content)["data"]

    enc_result = np.array([[[[decode_ciphertext(value) for value in row]
                             for row in column]
                            for column in layer]
                           for layer in enc_result])

    print("DEBUG: Print answer decrypted...")
    dec_result = decrypt_matrix(HE, enc_result)
    print(dec_result)
    #
    # # EXTRA DEBUG TO CHECK THE RESULTS
    # print("DEBUG: Plain results...")
    # plain_net = torch.load("./mnist.pt")
    # plain_net.eval()
    #
    # plain_net = plain_net[0:4]
    # results_plain = plain_net(torch.tensor(input_image))
    # print(results_plain)


if __name__ == '__main__':
    main()
