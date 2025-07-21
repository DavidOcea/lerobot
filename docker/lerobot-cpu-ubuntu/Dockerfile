# Configure image
ARG PYTHON_VERSION=3.10
FROM ubuntu:22.04


# Configure environment variables
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    libprotobuf-dev \
    protobuf-compiler \
    curl \
    wget \
    sudo \
    lsb-release \
    gnupg \
    gpg-agent \
    software-properties-common \
    speech-dispatcher libgeos-dev \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
apt-get update && apt-get install -y gcc-13 g++-13 && \
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 --slave /usr/bin/g++ g++ /usr/bin/g++-13 && \
apt-get clean && rm -rf /var/lib/apt/lists/*
# Clone repository and install LeRobot in a single layer
COPY . /lerobot
WORKDIR /lerobot
RUN /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[test, aloha, xarm, pusht, dynamixel]" \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Execute in bash shell rather than python
CMD ["/bin/bash"]
