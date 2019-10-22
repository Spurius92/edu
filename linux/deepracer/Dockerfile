FROM ubuntu:18.04

RUN echo 'Installing docker inside docker'
RUN apt-get update && \
    apt-get install -y --no-install-recommends\
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
    apt-key fingerprint 0EBFCD88 &&\
    add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable" \
    apt-get update &&\
    apt-get install docker-ce docker-ce-cli containerd.io &&\
    apt-get autoremove -y && \
    # remove apt cache from image
    apt-get clean all && \
    echo 'Finished!' &&\
    curl -L https://github.com/docker/compose/releases/download/1.24.1/docker-compose-`uname -s`-`uname -m` -o     /usr/local/bin/docker-compose &&\
    chmod +x /usr/local/bin/docker-compose &&\
    echo 'Verify installation. Version of docker compose is:' &&\
    docker-compose --version &&\
    groupadd docker &&\
    usermod -aG docker $USER &&\
    # Log out and log back in so that your group membership is re-evaluated.
    # And configure Docker to start on boot.
    # systemctl enable docker &&\
    # distribution=$(. /etc/os-release;echo $ID$VERSION_ID) &&\
    # curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - &&\
    # curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list &&\
    # sudo apt-get update &&\
    # sudo apt-get install nvidia-docker2 &&\
    # sudo pkill -SIGHUP dockerd &&\
    # sudo add-apt-repository ppa:graphics-drivers &&\
    # apt-get update &&\
    # apt install nvidia-driver-410 && \
    # sudo reboot  &&\
    # NOTE: 410 is a driver version that is compatible with the GPU I selected on the Nvidia website.
    # Verify the driver installation:
    # nvidia-smi &&\
    # nvcc --version  &&\