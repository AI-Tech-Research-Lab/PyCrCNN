import tempfile
import zipfile
import os

import jsonpickle
from Pyfhel.PyCtxt import PyCtxt
from flask import Flask, request
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
    directory = tempfile.TemporaryDirectory()

    parameters = jsonpickle.decode(request.files["encryption_parameters"].read())
    request.files["input"].save(directory.name + "/" + "input.zip")
    zf = zipfile.ZipFile(directory.name + "/" + 'input.zip', mode='r')
    zf.extractall(directory.name)


    enc_image = np.array([[list([[PyCtxt() for i in range(0, 28)] for j in range(0, 28)])]])

    for row in range(0, len(enc_image[0][0])):
        for col in range(0, len(enc_image[0][0][row])):
            name = str(row) + "-" + str(col)
            enc_image[0][0][row][col].load(directory.name + "/" + name, "float")

    print(enc_image.shape)
    print(enc_image[0][0][0][0])

    result = perform_computation(parameters, enc_image)

    return 'OK'
    #return jsonpickle.encode(result)



if __name__ == '__main__':
    app.run(debug=True)
