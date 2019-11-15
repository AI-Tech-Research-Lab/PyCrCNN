import glob
import os
import tempfile
import zipfile

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
        - Create a zip file containing all the ciphertext in the matrix, saved one by one in a file with a meaningful
          name, i.e "0-1-2-3" is the file containing the ciphertext of image 0, layer 1, row 2, column 3
        - Send the post request with the parameters and the zip file
        - After receiving the response, do the opposite of pass 4, i.e de-serialize the answer zip file in a
          numpy matrix of ciphertexts
        - After the result is reconstructed in a suitable numpy array of ciphertext, decrypt it to obtain the plain
          result
        - OPTIONAL: print both the result and the plain result, obtained doing the same computation locally to ensure
          the result is correct (for debug purposes)
    """

    request_temp_dir = tempfile.TemporaryDirectory()
    answer_temp_dir = tempfile.TemporaryDirectory()
    zip_temp_dir = tempfile.TemporaryDirectory()

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

    print("DEBUG: Zipping...")
    zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, 'input.zip'), mode='w')

    for image in range(0, len(enc_image)):
        for layer in range(0, len(enc_image[image])):
            for row in range(0, len(enc_image[image][layer])):
                for col in range(0, len(enc_image[image][layer][row])):
                    name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                    enc_image[image][layer][row][col].save(request_temp_dir.name + "/" + name)
                    zf.write(os.path.join(request_temp_dir.name, name), compress_type=zipfile.ZIP_DEFLATED, arcname=name)

    zf.close()
    print("DEBUG: Zip size=", os.path.getsize(os.path.join(zip_temp_dir.name, 'input.zip'))/(1024*1024), "MB")

    files = {
        'parameters': open("./parameters.json", "rb"),
        'input': open(os.path.join(zip_temp_dir.name, 'input.zip'), 'rb')
    }

    print("DEBUG: Sending the request...")
    response = requests.post(address, files=files)

    print("DEBUG: Response received, saving...")
    open(os.path.join(zip_temp_dir.name, 'answer.zip'), 'wb').write(response.content)

    print("DEBUG: Extracting response...")
    zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, 'answer.zip'), mode='r')
    zf.extractall(answer_temp_dir.name)

    print("DEBUG: Reconstructing answer...")
    answer_files = sorted([os.path.basename(f) for f in glob.glob(answer_temp_dir.name + "/*")])
    numbers = [file.split("-") for file in answer_files]
    images, layers, rows, columns = max([[int(x) + 1 for x in item] for item in numbers])

    result = np.array([[[[HE.encryptFrac(0) for i in range(0, columns)] for j in range(0, rows)]
                        for k in range(0, layers)] for z in range(0, images)])

    for image in range(0, images):
        for layer in range(0, layers):
            for row in range(0, rows):
                for col in range(0, columns):
                    name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                    result[image][layer][row][col].load(os.path.join(answer_temp_dir.name, name), "float")

    print("DEBUG: Print answer decrypted...")
    dec_result = decrypt_matrix(HE, result)
    print(dec_result)

    # EXTRA DEBUG TO CHECK THE RESULTS
    print("DEBUG: Plain results...")
    plain_net = torch.load("./mnist.pt")
    plain_net.eval()

    plain_net = plain_net[0:4]
    results_plain = plain_net(torch.tensor(input_image))
    print(results_plain)


if __name__ == '__main__':
    main()
