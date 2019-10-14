from Pyfhel import Pyfhel
from pycrcnn.net_builder import pytorch_net as pytorch
from pycrcnn.net_builder.encoded_net_builder import test_net

plain_net = pytorch.load_net()

data_loader = pytorch.new_batch(1)
my_batch = next(iter(data_loader))
my_images, labels = my_batch

values = []
results = []

value = int(input("Enter a number for p: "))
while value != 0:
    values.append(value)
    value = int(input("Enter a number: "))

debug = str(input("Debug? y/N: "))
rencry_position = str(input("Where to put rencryption? "))

print("-----------------------")

if debug == "y":
    debug = True
else:
    debug = False

for i in range(0, len(values)):
    HE = Pyfhel()
    HE.contextGen(p=values[i], m=2048, base=2)
    HE.keyGen()
    HE.relinKeyGen(20, 5)
    results.append((values[i], test_net(HE, plain_net, my_images, rencry_position, debug)))

print("-----------------------")
print("Tested values: ")
for i in range(0, len(results)):
    print(results[i][0])

print("-----------------------")

print("Obtained errors: ")
for i in range(0, len(results)):
    print(results[i][1])

print("-----------------------")
