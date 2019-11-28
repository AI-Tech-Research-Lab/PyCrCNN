import base64
import multiprocessing
import tempfile

import jsonpickle
from Pyfhel import Pyfhel
from flask import Flask, request
import numpy as np

from pycrcnn.network.server.executer import perform_computation

app = Flask(__name__)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/compute', methods=['POST'])
def handle_request():
    """
    This function receives a POST request with a JSON containing:
        - parameters
        - input

    Parameters contains both the encryption parameters requested, as p, m, sec level and base and the model requested,
    with the layers on which the computation is requested (for example: MNIST, layers from 0 to 3)

    Input contains the array of encoded PyCtxt. In this context, encoded means they are the dump of every ciphertext's
    file encoded in b64.

    Due to limitations in the serialization of Pyfhel objects, the following algorithm is executed:

        - Reconstruct the encrypted input, decoding every b64 value in the array data
        - After the image is reconstructed in a suitable numpy array of ciphertext, perform the encrypted computation
          with the layers encoded with the aforementioned parameters
        - Once the result is ready, serialize all the values of the answer in b64
        - Answer the post request with the JSON containing those values
    """

    def compute(payload, return_dict):
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

        payload = jsonpickle.decode(payload)
        encryption_parameters = payload["encryption_parameters"]
        data = payload["data"]

        HE = Pyfhel()
        HE.contextGen(m=encryption_parameters[0]["m"],
                      p=encryption_parameters[0]["p"],
                      sec=encryption_parameters[0]["sec"],
                      base=encryption_parameters[0]["base"])
        HE.keyGen()
        HE.relinKeyGen(20, 5)

        enc_image = np.array([[[[decode_ciphertext(value) for value in row]
                    for row in column]
                    for column in layer]
                    for layer in data])

        result = perform_computation(HE, enc_image, payload["net"], payload["layers"])

        to_encode = [[[[encode_ciphertext(value) for value in row]
                    for row in column]
                    for column in layer]
                    for layer in result]

        answer = {
            "data": to_encode
        }
        return_dict["answer"] = jsonpickle.encode(answer)


    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=compute, args=(request.json, return_dict))
    p.start()
    p.join()

    response = app.response_class(
        response=return_dict["answer"],
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
