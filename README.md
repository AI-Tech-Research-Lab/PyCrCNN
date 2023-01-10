# PyCrCNN
PyCrCNN is the implementation of a privacy-respectful Machine Learning as a Service (MLaaS) which use Homomorphic Encryption (HE).
The application has been tailored on Convolutional Neural Networks (CNNs).

PyCrCNN has been introduced in the [paper](https://arxiv.org/pdf/2003.13541.pdf) "A Privacy-Preserving Distributed Architecture for Deep-Learning-as-a-Service".

**For an introduction to Homomorphic Encryption, check out our [new work](https://github.com/AlexMV12/Introduction-to-BFV-HE-ML)!**

*Disclaimer: we are working on the topic, so, expect changes in this repo.*

## Introduction
PyCrCNN is a client/server application in which the server can run a Convolutional Neural Network on some data coming from a client.
The peculiarity of this application is that Homomorphic Encryption is used: the data coming from the client (in this case, an image) is encrypted and the server doesn't have the keys to decrypt it.

Computation on encrypted data is still allowed by the mathematical properties of this kind of encryption which is, in fact, homomorphic with respect to addition and multiplication. This means that the models have to be approximated, in order to contain only additions and multiplications.

## External libraries used
You can install them using the provided `requirements.txt` file.
WARNING: to install [Pyfhel](https://github.com/ibarrond/Pyfhel) a small modification is needed (check this [issue](https://github.com/ibarrond/Pyfhel/issues/124)).
It is suggested to clone Pyfhel using the instructions on their repo (to correctly crone all the submodules), modify `pyproject.toml` to set the `SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT` to `OFF` and install from there with `pip install .`.

## How to use
Clone the repo, install the dependencies and try to run the notebook `HE-ML/HE-ML_CKKS.ipynb`. It will give you a fair introduction to the package.

### Docker
You can also use Docker in order to start a Jupyter Notebook with all the dependencies installed and correctly configured:
```bash
docker run --publish 8888:8888 alexmv12/pycrcnn
```
then, just open the link to the Jupyter notebook. 

## Package organization

- convolutional: Code for convolutional layers
- crypto: Code for crypyographic operations, like encrypting matrix, encoding matrix, etc
- functional: Code for functional operations like Square Layers, AvgPool layers, etc
- linear: Code for linear layers
- net_builder: Code for building encoded model, starting from a PyTorch model
- network: Client and server code
- parameters_tester: Code to let user test the encryption parameters, given a model and such parameters. Statistics will be put in output.
- tests

## Functionalities

### Encoded layers
Encoded layers are layers with the same characteristics of a normal PyTorch layer, but their weights have been encoded: this means that they are good to work on encrypted data.

In fact, in a normal forward on a normal CNN, computations between numbers are of the type:

    (Float, Float) -> Float
    
In an HE environment, computations between numbers are of the type:

    (EncryptedFrac, EncodedFrac) -> EncryptedFrac


In general, layers have been made with an eye on PyTorch definition: the reason is to be quite similar in the use to the correspondent PyTorch objects, without many differences.
For example, an encoded convolutional layer can be built with:


```python
ConvolutionalLayer(HE, 
                   weights=torchLayer.weight.detach().numpy(),
                   stride=torchLayer.stride, 
                   padding=torchLayer.padding, 
                   bias=torchLayer.bias)
```

The weights will be automatically encoded in Encoded Fractionals.

Furthermore, like in PyTorch, layers are callable objects: that means that, if we have an image (of the form of a numpy.array of EncryptedFrac), we can call the forward function off the layer just with


```python
results = encodedLayer(encrypted_image)
```

### Net builder
The code provided in the net_builder subpackage can retrieve a PyTorch model with some restrictions and build an encoded model with the same layers/weights: that is, an equivalent model able to work on encrypted data.

PyTorch model have to respect these constraints:
- It must be a [Sequential model](https://pytorch.org/docs/stable/nn.html#sequential)
- It must have extension .pt / .pth
- It must have been saved in PyTorch with the [save() function](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model).

If they are satisfied, the model can be built with:


```python
build_from_pytorch(HE, net, rencrypt_positions=[])
```

Where
- HE is a Pyfhel object
- net is the PyTorch object, loaded with the load() function
- rencrypt_positions can be used to put some rencryption layers in the resulting encoded model, i.e layers in which the input image is re-encrypted in order to restore the Noise Budget to the original level: in a real scenario, the server should send the partial results back to the client which will re-encrypt them and re-send them to the server to continue.

The function will return an ordered list of encoded layers.

### Parameter tester

The parameter tester loads a set of encryption parameters in JSON, a PyTorch model, and tests that model against its encoded corrispective on an image.
An example of input may be:


```python
{
  "encryption_parameters" : [
    {
      "m": 2048,
      "p": 13513511,
      "sec": 128,
      "base": 2,
      "rencrypt_positions": [ ]
    }
  ],
  "debug" : "y"
}
```

After the computation, a JSON results will be procuded: it will contain variouos information like the maximum error (that is the maximum difference between a value in the plain case and its corrispective encrypted value), the average error, the noise budget consumption etc.

### REST Client/Server
Built with flask, the server proides a REST endpoint to let client send requests for computations.
![image.png](attachment:image.png)

An example of client parameters is:


```python
{
  "address" : "http://127.0.0.1:5000/compute",
  "encryption_parameters" : [
    {
      "m": 2048,
      "p": 13513511,
      "sec": 128,
      "base": 2
    }
  ],
  "net" : "MNIST",
  "layers" : [0, 1, 2, 3]
}
```

In this example, the client is asking to forward the image on the net called "MNIST", only on layers from 0 to 3.
The client has encrypted the image with such parameters: the server will encode the model with them as well.
