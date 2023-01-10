FROM ubuntu:lunar

RUN apt update
RUN apt install -y python3 python3-pip cmake git

RUN git clone https://github.com/AlexMV12/PyCrCNN
RUN pip install -r PyCrCNN/requirements.txt
RUN pip install jupyter

RUN git clone --recursive https://github.com/ibarrond/Pyfhel.git
WORKDIR "/Pyfhel"
RUN sed -i "s/TRANSPARENT_CIPHERTEXT='ON'/TRANSPARENT_CIPHERTEXT='OFF'/" pyproject.toml
RUN pip install .

RUN pip install --upgrade numpy

WORKDIR "/PyCrCNN"

CMD jupyter notebook --ip 0.0.0.0 --allow-root --no-browser


