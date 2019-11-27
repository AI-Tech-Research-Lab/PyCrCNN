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

    def compute(parameters, input_zip_file):

        parameters = jsonpickle.decode(parameters.read())

        request_temp_dir = tempfile.TemporaryDirectory()
        answer_temp_dir = tempfile.TemporaryDirectory()
        zip_temp_dir = tempfile.TemporaryDirectory()

        input_zip_file.save(os.path.join(zip_temp_dir.name, "input.zip"))
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

        zip_temp_dir = tempfile.TemporaryDirectory()
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

        return jsonpickle.encode(open(os.path.join(zip_temp_dir.name, "answer.zip"), "rb").read())

    response = app.response_class(
        response=compute(request.files["parameters"], request.files["input"]),
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
