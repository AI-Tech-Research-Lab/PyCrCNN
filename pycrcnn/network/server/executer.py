import jsonpickle

from pycrcnn.net_builder.encoded_net_builder import build_from_pytorch


def perform_computation(HE, enc_images, net, layers):
    # Or whetever the plain net is
    if net == "MNIST":
        with open("./mnist.json", "r") as f:
            plain_net = jsonpickle.decode(f.read())

    # Choose how many layers encode
    plain_net = plain_net[min(layers):max(layers)+1]
    encoded_net = build_from_pytorch(HE, plain_net)

    for layer in encoded_net:
        enc_images = layer(enc_images)

    return enc_images

