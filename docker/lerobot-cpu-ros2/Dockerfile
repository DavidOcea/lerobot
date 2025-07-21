# Configure image
ARG PYTHON_VERSION=3.10
FROM ubuntu:22.04


# Configure environment variables
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"
# Set locale for ROS2
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install dependencies, ROS2, and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic build and system tools
    build-essential cmake git curl wget sudo lsb-release gnupg gpg-agent software-properties-common \
    # Graphics and simulation dependencies
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    # Other dependencies for LeRobot
    libprotobuf-dev protobuf-compiler speech-dispatcher libgeos-dev \
    # Python dependencies
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


    # Locale dependency for ROS2
    RUN apt-get update && apt-get install -y --no-install-recommends \    
    locales \
    && locale-gen en_US.UTF-8 \
    # Add the ROS2 GPG key
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    # Add the ROS2 repository
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    # Update package list again after adding new repo and install ROS2
    && apt-get update && apt-get install -y --no-install-recommends \
    # Install ROS2 Humble Desktop and Python build tools
    ros-humble-desktop python3-colcon-common-extensions \
    # Create Python symlink and venv
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    # Create a venv that can access system-site-packages (for ROS2 libraries)
    && python -m venv --system-site-packages /opt/venv \
    # Clean up
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    # Source ROS2 and the venv in bashrc for interactive sessions
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Install a newer GCC version (as in the original Dockerfile)
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