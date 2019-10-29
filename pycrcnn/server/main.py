import tempfile
import zipfile
import os

import jsonpickle
from Pyfhel import Pyfhel
from flask import Flask, request, send_file
import numpy as np

from pycrcnn.server.executer import perform_computation

app = Flask(__name__)

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         static_file = request.files['the_file']
#         # here you can send this static_file to a storage service
#         # or save it permanently to the file system
#         static_file.save('/var/www/uploads/profilephoto.png')


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/compute', methods=['POST'])
def handle_request():
    # if request.headers['Content-Type'] == 'application/json':
    print("DEBUG: Request arrived...")
    request_temp_dir = tempfile.TemporaryDirectory()
    answer_temp_dir = tempfile.TemporaryDirectory()


    print("DEBUG: Unzipping and collecting encryption parameters...")
    parameters = jsonpickle.decode(request.files["encryption_parameters"].read())
    request.files["input"].save(request_temp_dir.name + "/" + "input.zip")
    zf = zipfile.ZipFile(request_temp_dir.name + "/" + 'input.zip', mode='r')
    zf.extractall(request_temp_dir.name)

    HE = Pyfhel()
    HE.contextGen(m=parameters["encryption_parameters"][0]["m"],
                  p=parameters["encryption_parameters"][0]["p"],
                  sec=parameters["encryption_parameters"][0]["sec"],
                  base=parameters["encryption_parameters"][0]["base"])
    HE.keyGen()
    HE.relinKeyGen(20, 5)

    print("DEBUG: Reconstructing the image...")
    enc_image = np.array([[list([[HE.encryptFrac(0) for i in range(0, 28)] for j in range(0, 28)])]])

    for image in range(0, len(enc_image)):
        for layer in range(0, len(enc_image[image])):
            for row in range(0, len(enc_image[image][layer])):
                for col in range(0, len(enc_image[image][layer][row])):
                    name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                    enc_image[image][layer][row][col].load(request_temp_dir.name + "/" + name, "float")

    print("DEBUG: Perform the computation on the image...")
    result = perform_computation(HE, enc_image)

    print("DEBUG: Zipping the response...")
    zf = zipfile.ZipFile(answer_temp_dir.name + "/" + 'answer.zip', mode='w')

    for image in range(0, len(result)):
        for layer in range(0, len(result[image])):
            for row in range(0, len(result[image][layer])):
                for col in range(0, len(result[image][layer][row])):
                    name = str(image) + "-" + str(layer) + "-" + str(row) + "-" + str(col)
                    result[image][layer][row][col].save(answer_temp_dir.name + "/" + name)
                    zf.write(answer_temp_dir.name + "/" + name, arcname=name)

    zf.close()

    print("DEBUG: Sending the response...")

    return send_file(answer_temp_dir.name + "/" + 'answer.zip', as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
