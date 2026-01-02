FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Basis-Tools + C++-Deps
RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    python3-pip \
    libeigen3-dev libboost-all-dev \
    libjpeg-dev libgif-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /root

# Franka-Stack über Debian-Pakete
RUN apt-get update && apt-get install -y \
    ros-noetic-libfranka \
    && rm -rf /var/lib/apt/lists/*

# -------- catkin workspace + franka_ros (ohne Gazebo) --------
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws

RUN git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros && \
    rm -rf src/franka_ros/franka_gazebo

# ROS-Core-Dependencies für franka_ros
RUN apt-get update && apt-get install -y \
    ros-noetic-catkin \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-actionlib-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-std-srvs \
    ros-noetic-rostest \
    ros-noetic-rosunit \
    && rm -rf /var/lib/apt/lists/*

# ros_control-Abhängigkeiten für franka_hw
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-combined-robot-hw \
    && rm -rf /var/lib/apt/lists/*

# franka_ros bauen
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && catkin_make"

# -------- TIBA kopieren (NEU) --------
COPY tiba /root/catkin_ws/src/tiba

# -------- Python-Tools für Tiba (kompatible Versionen) --------
RUN pip install \
      'roboticstoolbox-python>=0.11.0' \
      'spatialmath-python>=0.10.0' \
      plotly networkx tqdm svgpathtools \
      pycairo moderngl pyglet \
      'imgui[pyglet]' easydict xdialog \
      drawsvg glfw rumore numpy scipy matplotlib

# py5canvas (GUI-Framework)
RUN pip install py5canvas

# -------- TIBA bauen & installieren (NEU) --------
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /root/catkin_ws && \
    catkin_make && \
    source devel/setup.bash && \
    cd src/tiba && \
    pip install -e ."

# Setup ROS Environment automatically
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# Default: interaktive Shell mit ROS setup
CMD ["/bin/bash"]
