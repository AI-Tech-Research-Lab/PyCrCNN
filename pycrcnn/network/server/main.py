import glob
import tempfile
import time
import zipfile
import os
from multiprocessing import Process

import jsonpickle
import torch
from Pyfhel import Pyfhel
from flask import Flask, request, send_file
import numpy as np

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch
from pycrcnn.network.server.executer import perform_computation

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/compute', methods=['POST'])
def handle_request():
    """
    This function receives a POST request with two files:
        - parameters: JSON file
        - input: zip file

    Parameters contains both the encryption parameters requested, as p, m, sec level and base and the model requested,
    with the layers on which the computation is requested (for example: MNIST, layers from 0 to 3)

    Due to limitations in the serialization of Pyfhel objects, the following algorithm is executed:

        - Extract all the files in the input file in a temporary directory
        - Reconstruct the encrypted image in a numpy array: every extracted file matches a position in the matrix,
          i.e "0-1-2-3" is the file containing the ciphertext of image 0, layer 1, row 2, column 3
        - After the image is reconstructed in a suitable numpy array of ciphertext, perform the encrypted computation
          with the layers encoded with the aforementioned parameters
        - Once the result is ready, serialize that in a zip file doing the opposite process of point 1
        - Answer the post request with this file
    """

    def compute_debug():
        print("DEBUG: Request arrived...")
        print("DEBUG: Unzipping and collecting encryption parameters...")
        start_time = time.process_time()
        parameters = jsonpickle.decode(request.files["parameters"].read())
        request.files["input"].save(os.path.join(zip_temp_dir.name, "input.zip"))
        zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, "input.zip"), mode='r')
        zf.extractall(request_temp_dir.name)

        time_unzip = time.process_time() - start_time

        HE = Pyfhel()
        HE.contextGen(m=parameters["encryption_parameters"][0]["m"],
                      p=parameters["encryption_parameters"][0]["p"],
                      sec=parameters["encryption_parameters"][0]["sec"],
                      base=parameters["encryption_parameters"][0]["base"])
        HE.keyGen()
        HE.relinKeyGen(20, 5)

        print("DEBUG: Reconstructing the image...")
        input_files = sorted([os.path.basename(f) for f in glob.glob(request_temp_dir.name + "/*")])
        numbers = [file.split("-") for file in input_files]
        images, layers, rows, columns = max([[int(x)+1 for x in item] for item in numbers])

        enc_image = np.array([[[[HE.encryptFrac(0) for i in range(0, columns)] for j in range(0, rows)]
                            for k in range(0, layers)] for z in range(0, images)])

        def populate_enc_image(image, layer, row, col):
            name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
            enc_image[image][layer][row][col].load(os.path.join(request_temp_dir.name, name), "float")

        [[[[populate_enc_image(image, layer, row, col) for col in range(0, columns)] for
          row in range (0, rows)] for layer in range(0, layers)] for image in range(0, images)]

        time_reconstructing = time.process_time() - time_unzip

        print("DEBUG: Perform the computation on the image...")
        layers = parameters["layers"]

        if parameters["net"] == "MNIST":
            plain_net = torch.load("./mnist.pt")
            plain_net.eval()

        # Choose how many layers encode
        plain_net = plain_net[min(layers):max(layers) + 1]
        encoded_net = build_from_pytorch(HE, plain_net)

        time_encoding = time.process_time() - time_unzip

        for layer in encoded_net:
            enc_image = layer(enc_image)
        result = enc_image

        time_computation = time.process_time() - time_encoding

        print("DEBUG: Zipping the response...")
        zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, "answer.zip"), mode='w')

        for image in range(0, len(result)):
            for layer in range(0, len(result[image])):
                for row in range(0, len(result[image][layer])):
                    for col in range(0, len(result[image][layer][row])):
                        name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                        result[image][layer][row][col].save(os.path.join(answer_temp_dir.name, name))
                        zf.write(os.path.join(answer_temp_dir.name, name), compress_type=zipfile.ZIP_DEFLATED,
                                 arcname=name)
        zf.close()
        time_zipping = time.process_time() - time_computation

        print("DEBUG: Sending the response...")
        total_time = (time_unzip + time_reconstructing + time_encoding + time_computation + time_zipping) / 100
        print("Time to unzip: ", '{:6.3f}'.format(time_unzip), " seconds, % = ", '{:6.3f}'.format(time_unzip / total_time))
        print("Time to reconstruct image: ", '{:6.3f}'.format(time_reconstructing), " seconds, % = ", '{:6.3f}'.format(time_reconstructing / total_time))
        print("Time to encode layers: ", '{:6.3f}'.format(time_encoding), " seconds, % = ", '{:6.3f}'.format(time_encoding / total_time))
        print("Time to compute: ", '{:6.3f}'.format(time_computation), " seconds, % = ", '{:6.3f}'.format(time_computation / total_time))
        print("Time to zip result: ", '{:6.3f}'.format(time_zipping), " seconds, % = ", '{:6.3f}'.format(time_zipping / total_time))

    def compute():

        parameters = jsonpickle.decode(request.files["parameters"].read())
        request.files["input"].save(os.path.join(zip_temp_dir.name, "input.zip"))
        zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, "input.zip"), mode='r')
        zf.extractall(request_temp_dir.name)

        HE = Pyfhel()
        HE.contextGen(m=parameters["encryption_parameters"][0]["m"],
                      p=parameters["encryption_parameters"][0]["p"],
                      sec=parameters["encryption_parameters"][0]["sec"],
                      base=parameters["encryption_parameters"][0]["base"])
        HE.keyGen()
        HE.relinKeyGen(20, 5)

        input_files = sorted([os.path.basename(f) for f in glob.glob(request_temp_dir.name + "/*")])
        numbers = [file.split("-") for file in input_files]
        images, layers, rows, columns = max([[int(x)+1 for x in item] for item in numbers])

        enc_image = np.array([[[[HE.encryptFrac(0) for i in range(0, columns)] for j in range(0, rows)]
                            for k in range(0, layers)] for z in range(0, images)])

        for image in range(0, images):
            for layer in range(0, layers):
                for row in range(0, rows):
                    for col in range(0, columns):
                        name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                        enc_image[image][layer][row][col].load(os.path.join(request_temp_dir.name, name), "float")

        result = perform_computation(HE, enc_image, parameters["net"], parameters["layers"])

        zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, "answer.zip"), mode='w')

        for image in range(0, len(result)):
            for layer in range(0, len(result[image])):
                for row in range(0, len(result[image][layer])):
                    for col in range(0, len(result[image][layer][row])):
                        name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                        result[image][layer][row][col].save(os.path.join(answer_temp_dir.name, name))
                        zf.write(os.path.join(answer_temp_dir.name, name), compress_type=zipfile.ZIP_DEFLATED, arcname=name)

        zf.close()

    request_temp_dir = tempfile.TemporaryDirectory()
    answer_temp_dir = tempfile.TemporaryDirectory()
    zip_temp_dir = tempfile.TemporaryDirectory()

    # Executes the computation on Pyfhel objects in a separate process to avoid memory unreferenced
    if app.debug:
        p = Process(target=compute_debug)
    else:
        p = Process(target=compute)
    p.start()
    p.join()

    return send_file(os.path.join(zip_temp_dir.name, "answer.zip"), as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
