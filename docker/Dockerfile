FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git nano less curl wget gnupg2 \
        lsb-release ca-certificates iproute2 iputils-ping \
        python3-pip python3-colcon-common-extensions python3-rosdep \
        mesa-utils libgl1-mesa-dri libglx-mesa0 libegl-mesa0 \
        libgles2 libglu1-mesa \
        libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 \
        x11-utils x11-apps \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-va-drivers mesa-vulkan-drivers libvulkan1 \
        va-driver-all vainfo \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    wget -qO- https://cyberbotics.com/Cyberbotics.asc \
        | gpg --dearmor -o /etc/apt/keyrings/cyberbotics.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/cyberbotics.gpg] \
https://cyberbotics.com/debian binary-amd64/" \
        > /etc/apt/sources.list.d/cyberbotics.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        webots \
        ros-humble-webots-ros2-driver \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2 libnss3 libxslt1.1 libssh-4 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rviz2 \
        ros-humble-rqt-common-plugins \
        ros-humble-tf2-ros ros-humble-tf2-tools \
        ros-humble-tf-transformations \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-topic-tools \
        ros-humble-teleop-twist-keyboard \
        ros-humble-robot-localization \
        ros-humble-imu-filter-madgwick ros-humble-imu-tools \
        ros-humble-slam-toolbox \
        ros-humble-navigation2 ros-humble-nav2-bringup \
        ros-humble-nav2-smac-planner \
        ros-humble-nav2-regulated-pure-pursuit-controller \
        ros-humble-nav2-map-server \
        ros-humble-ackermann-msgs \
        ros-humble-rosbag2 ros-humble-rosbag2-storage-mcap \
        ros-humble-image-transport ros-humble-compressed-image-transport \
        ros-humble-nav2-rviz-plugins \
    && rm -rf /var/lib/apt/lists/*

# Nav2 + perception stack (ros2_ws/src/offroad_nav2): stereo depth, vision_msgs, collision monitor, PyTorch/CUDA.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-stereo-image-proc ros-humble-image-proc ros-humble-vision-msgs \
        ros-humble-nav2-collision-monitor ros-humble-nav2-velocity-smoother ros-humble-nav2-behavior-tree \
        ros-humble-xacro ros-humble-urdfdom ros-humble-cv-bridge python3-pytest \
    && rm -rf /var/lib/apt/lists/*
# Humble's cv_bridge is built against numpy 1.x, so numpy<2. The cu128 index is needed for Blackwell GPUs (sm_120).
RUN pip3 install --no-cache-dir "numpy<2" torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple
# torch's dependencies pull a setuptools that breaks colcon's ament_python builds on Ubuntu 22.04: restore the distro version.
RUN pip3 install --no-cache-dir "setuptools==59.6.0"
# ONNX Runtime GPU for the C++ perception node (headers + libs from the official release) and for tools/export_onnx.py (pip).
# ORT 1.23 wants CUDA 12 / cuDNN 9, while the torch wheel above brings CUDA 13: install the CUDA 12 runtime libs next to it
# (different sonames, no clash) and put them on the library path.
ARG ONNXRUNTIME_VERSION=1.23.2
RUN wget -qO- https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
        | tar xz -C /opt \
    && mv /opt/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION} /opt/onnxruntime
RUN pip3 install --no-cache-dir onnx onnxscript onnxruntime-gpu==${ONNXRUNTIME_VERSION} \
        nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 \
    && pip3 install --no-cache-dir "setuptools==59.6.0" \
    && (ls -d /usr/local/lib/python3.10/dist-packages/nvidia/*/lib; echo /opt/onnxruntime/lib) > /etc/ld.so.conf.d/onnxruntime-cuda12.conf \
    && ldconfig

# onnx / onnxruntime-gpu pull numpy 2.x, which breaks Humble's cv_bridge / OpenCV (built for numpy 1.x): pin it back.
RUN pip3 install --no-cache-dir "numpy<2" "setuptools==59.6.0"

# Marker used by `make` to rebuild images that predate the Nav2 + perception layers.
LABEL vesta.nav="2"

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV ROS_DOMAIN_ID=0
ENV WEBOTS_HOME=/usr/local/webots
ENV PATH="${WEBOTS_HOME}:${PATH}"

ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}"


RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo '[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash' \
        >> /root/.bashrc && \
    echo "PS1='${debian_chroot:+($debian_chroot)}\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;33m\]\w\[\033[00m\]\$ '" \
        >> /root/.bashrc && \
    echo "USERNAME=root" \
        >> /root/.bashrc && \
    echo "USER=root" >> /root/.bashrc

WORKDIR /ros2_ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
