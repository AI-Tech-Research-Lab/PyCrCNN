import tempfile
import zipfile

import jsonpickle
import numpy as np

import requests
from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from pycrcnn.crypto.crypto import decrypt_matrix, encrypt_matrix

directory = tempfile.TemporaryDirectory()

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

enc_image = encrypt_matrix(HE, input_image)

zf = zipfile.ZipFile(directory.name + "/" + 'input.zip', mode='w')

result = np.array([[PyCtxt() for column in row] for row in enc_image])

for row in range(0, len(input_image[0][0])):
    for col in range(0, len(input_image[0][0][row])):
        name = str(row) + "-" + str(col)
        enc_image[0][0][row][col].save(directory.name + "/" + name)
        zf.write(directory.name + "/" + name, arcname=name)

zf.close()


files = {
     'encryption_parameters': open("./encryption_parameters.json", "rb"),
     'input': open(directory.name + '/input.zip', 'rb')
}


response = requests.post('http://127.0.0.1:5000/compute', files=files)

print(response.content)