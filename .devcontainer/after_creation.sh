#/bin/bash

# 1. Clone the Pyfhel repository
git clone --recursive https://github.com/ibarrond/Pyfhel.git ~/Pyfhel

# 2. Fix the `throw_on_transparent_ciphertext` flag
sed -i "s/SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT='ON'/SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT='OFF'/" ~/Pyfhel/pyproject.toml

# 3. Better to use Python 3.10.
pyenv install 3.10
pyenv global 3.10

# 4. Install Pyfhel
python -m venv venv
source venv/bin/activate
pip install ~/Pyfhel/

# 5. Install also TenSEAL
pip install TenSEAL

# 6. Install Torch (CPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. Install PyCrCNN
pip install .
