import glob
import tempfile
import zipfile
import os

import jsonpickle
from Pyfhel import Pyfhel
from flask import Flask, request, send_file
import numpy as np

from pycrcnn.server.executer import perform_computation

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/compute', methods=['POST'])
def handle_request():

    print("DEBUG: Request arrived...")
    request_temp_dir = tempfile.TemporaryDirectory()
    answer_temp_dir = tempfile.TemporaryDirectory()
    zip_temp_dir = tempfile.TemporaryDirectory()

    print("DEBUG: Unzipping and collecting encryption parameters...")
    parameters = jsonpickle.decode(request.files["encryption_parameters"].read())
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

    print("DEBUG: Reconstructing the image...")
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

    print("DEBUG: Perform the computation on the image...")
    result = perform_computation(HE, enc_image)

    print("DEBUG: Zipping the response...")
    zf = zipfile.ZipFile(os.path.join(zip_temp_dir.name, "answer.zip"), mode='w')

    for image in range(0, len(result)):
        for layer in range(0, len(result[image])):
            for row in range(0, len(result[image][layer])):
                for col in range(0, len(result[image][layer][row])):
                    name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                    result[image][layer][row][col].save(answer_temp_dir.name + "/" + name)
                    zf.write(os.path.join(answer_temp_dir.name, name), compress_type=zipfile.ZIP_DEFLATED, arcname=name)

    zf.close()

    print("DEBUG: Sending the response...")

    return send_file(os.path.join(zip_temp_dir.name, "answer.zip"), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
