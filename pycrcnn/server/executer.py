import torch
from Pyfhel import Pyfhel

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


def perform_computation(HE, enc_images):

        # Or whetever the plain net is
        plain_net = torch.load("./mnist.pt")
        plain_net.eval()

        # Choose how many layers encode
        encoded_net = build_from_pytorch(HE, plain_net[0:4])

        for layer in encoded_net:
            enc_images = layer(enc_images)

        return enc_images


# 1) Ipoteticamente la rete è già presente in chiaro sul server, caricabile da file
# 2) Ricevo una richiesta con dei parametri di encryption e un'immagine criptata (?), tutto in JSON
# 3) Encodo la rete
# 4) Computo sui primi tot layer
# 5) Restituisco indietro il risultato

