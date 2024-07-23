FROM archlinux:latest

# Populate keys
RUN pacman-key --init

# Install needed packages
RUN pacman --noconfirm -Syu git make cmake gcc sudo python python-setuptools python-pip pyenv

# Create a user with a specific UID and GID
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set the user
USER $USERNAME

# Enable pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Source the .bashrc
RUN /bin/bash -c "source ~/.bashrc"