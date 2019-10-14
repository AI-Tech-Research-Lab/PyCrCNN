class FlattenLayer:
    """
    A class used to represent a layer which performs a
    flattening operation.
    ...

    Attributes
    ----------

    length: int
        second dimension of the output matrix

    Methods
    -------
    __init__(self, length)
        Constructor of the layer.
    __call__(self, t)
        Taken an input tensor in the form
            [n_images, n_layers, y, x]
        Reshapes it in the form
            [n_images, length]
    """

    def __call__(self, image):
        dimension = image.shape
        return image.reshape(dimension[0], dimension[1]*dimension[2]*dimension[3])
