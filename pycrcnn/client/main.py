import glob
import os
import tempfile
import zipfile

import jsonpickle
import numpy as np

import requests
from Pyfhel import Pyfhel

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix


def main():
    request_temp_dir = tempfile.TemporaryDirectory()
    answer_temp_dir = tempfile.TemporaryDirectory()
    zip_temp_dir = tempfile.TemporaryDirectory()

    with open("./input_image.json", "rb") as f:
        input_image = jsonpickle.decode(f.read())

    with open("./encryption_parameters.json", "r") as f:
        encryption_parameters = jsonpickle.decode(f.read())["encryption_parameters"]

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
        'encryption_parameters': open("./encryption_parameters.json", "rb"),
        'input': open(os.path.join(zip_temp_dir.name, 'input.zip'), 'rb')
    }

    print("DEBUG: Sending the request...")
    response = requests.post('http://127.0.0.1:5000/compute', files=files)

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
    # print("DEBUG: Plain results...")
    # plain_net = torch.load("./mnist.pt")
    # plain_net.eval()
    #
    # plain_net = plain_net[0:4]
    # results_plain = plain_net(torch.tensor(input_image))
    # print(results_plain)


if __name__ == '__main__':
    main()
