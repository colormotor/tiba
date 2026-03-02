#!/usr/bin/env bash
set -e

# Basis-Tools

sudo apt-get update
sudo apt-get install -y \
  git curl gnupg lsb-release \
  build-essential cmake \
  python3-pip \
  libeigen3-dev libboost-all-dev \
  libjpeg-dev libgif-dev

python3 -m pip install --upgrade pip

# Miniforge + ros_env (Client-Setup wie auf dem Desktop)


# Miniforge installieren (falls noch nicht da)
if [ ! -d "$HOME/miniforge3" ]; then
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3-Linux-x86_64.sh
  chmod +x Miniforge3-Linux-x86_64.sh
  ./Miniforge3-Linux-x86_64.sh -b
  echo "source \"$HOME/miniforge3/etc/profile.d/conda.sh\"" >> "$HOME/.bashrc"
fi

# conda/mamba in diese Shell laden
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# mamba + Env
conda install -y conda-forge::mamba
mamba create -y -n ros_env python=3.9

# Env mit conda aktivieren
conda activate ros_env

# Channels
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults || true

# ROS + Tools im Env (wie auf dem Client)
mamba install -y ros-noetic-desktop
mamba install -y conda-forge::roboticstoolbox-python
mamba install -y plotly networkx tqdm svgpathtools
mamba install -y conda-forge::pycairo conda-forge::moderngl

# Python-Pakete (Client-Teil) IM ros_env
python -m pip install \
  trajectory_smoothing@git+https://github.com/colormotor/trajectory_smoothing.git \
  pyglet \
  "imgui[pyglet]" \
  easydict \
  xdialog \
  py5canvas==0.5.2 \
  glfw PyOpenGL imgui glcontext

# ilqr_planner in diesem Env bauen (für PyLQR)

conda activate ros_env
cd "$HOME"
if [ ! -d "$HOME/ilqr_planner" ]; then
  git clone https://github.com/colormotor/ilqr_planner.git
fi
cd "$HOME/ilqr_planner"
git submodule update --init --recursive

cd "$HOME/ilqr_planner/ilqr_planner"
sed -i '1ioption(BUILD_EXAMPLES "Build examples" OFF)' CMakeLists.txt

cd "$HOME/ilqr_planner/ilqr_planner/ilqr_planner/lib/tinyurdfparser"
sed -i '1ioption(BUILD_EXAMPLES "Build examples" OFF)' CMakeLists.txt
sed -i '2ioption(BUILD_TESTS "Build tests" OFF)' CMakeLists.txt
sed -i 's/add_subdirectory(examples)/if(BUILD_EXAMPLES)\n add_subdirectory(examples)\nendif()/g' CMakeLists.txt


./build.sh || echo "Warnung: ./build.sh von ilqr_planner ist fehlgeschlagen, bitte ggf. manuell prüfen."


# Systemlibs für OpenGL (für moderngl/py5canvas, falls benötigt)

sudo apt-get update
sudo apt-get install -y \
  libegl1 \
  libegl1-mesa \
  libgl1 \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  mesa-utils

sudo ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1  /usr/lib/x86_64-linux-gnu/libGL.so

# ROS Noetic + Franka (Server-Teil)


# ROS1 Repo hinzufügen (falls noch nicht vorhanden)
if [ ! -f /etc/apt/sources.list.d/ros1-latest.list ]; then
  sudo sh -c "echo 'deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main' > /etc/apt/sources.list.d/ros1-latest.list"
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
fi

sudo apt-get update

# ROS Noetic + libfranka + ROS-Dependencies (System-seitig, wie in der Doku)
sudo apt-get install -y \
  ros-noetic-desktop-full \
  ros-noetic-libfranka \
  ros-noetic-catkin \
  ros-noetic-roscpp \
  ros-noetic-std-msgs \
  ros-noetic-geometry-msgs \
  ros-noetic-actionlib-msgs \
  ros-noetic-visualization-msgs \
  ros-noetic-std-srvs \
  ros-noetic-rostest \
  ros-noetic-rosunit \
  ros-noetic-ros-control \
  ros-noetic-ros-controllers \
  ros-noetic-combined-robot-hw

# catkin workspace + franka_ros (ohne Gazebo)
mkdir -p "$HOME/catkin_ws/src"
cd "$HOME/catkin_ws"

if [ ! -d src/franka_ros ]; then
  git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros
fi

rm -rf src/franka_ros/franka_gazebo

# franka_ros bauen (nutzt /opt/ros/noetic)
source /opt/ros/noetic/setup.bash
cd "$HOME/catkin_ws"
catkin_make

# TIBA in catkin_ws/src/tiba integrieren und im ros_env installieren

cd "$HOME/catkin_ws/src"

# TIBA genau hierhin klonen: catkin_ws/src/tiba
if [ ! -d tiba ]; then
  git clone https://github.com/colormotor/tiba.git
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ros_env

# TIBA im ros_env installieren (editable)
cd "$HOME/catkin_ws/src/tiba"
python -m pip install -e .

# ROS-Umgebung automatisch setzen

if ! grep -q "source /opt/ros/noetic/setup.bash" "$HOME/.bashrc"; then
  echo "source /opt/ros/noetic/setup.bash" >> "$HOME/.bashrc"
fi

if ! grep -q "source \$HOME/catkin_ws/devel/setup.bash" "$HOME/.bashrc"; then
  echo "source \$HOME/catkin_ws/devel/setup.bash" >> "$HOME/.bashrc"
fi

echo
echo "Full Tiba-Setup fertig."

