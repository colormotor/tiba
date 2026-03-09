#!/usr/bin/env bash
set -e

# install git
sudo apt update
sudo apt install -y git

# install Miniforge if not existing
if [ ! -d "$HOME/miniforge3" ]; then
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  chmod +x Miniforge3-Linux-x86_64.sh
  ./Miniforge3-Linux-x86_64.sh -b
  echo "source $HOME/miniforge3/etc/profile.d/conda.sh" >> ~/.bashrc
  source ~/.bashrc
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"

# mamba + Env
conda install -y conda-forge::mamba
mamba create -y -n ros_env python=3.9
conda activate ros_env

# Channels
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults || true

# ROS + Tools
mamba install -y ros-noetic-desktop
mamba install -y conda-forge::roboticstoolbox-python
mamba install -y plotly networkx tqdm svgpathtools
mamba install -y conda-forge::pycairo conda-forge::moderngl

# Python-Packages
pip install \
  trajectory_smoothing@git+https://github.com/colormotor/trajectory_smoothing.git \
  pyglet \
  "imgui[pyglet]" \
  easydict \
  xdialog \
  py5canvas==0.5.2 \
  glfw PyOpenGL imgui glcontext

# TODO change path to path of tiba
cd "$HOME/Documents"

# Only clone repo if not existing yet
if [ ! -d "tiba" ]; then
  git clone https://github.com/colormotor/tiba.git
fi

cd tiba
pip install -e .

# ilqr_planner
cd "$HOME/Documents"
git clone https://github.com/colormotor/ilqr_planner.git || true
cd ilqr_planner
git submodule update --init --recursive
./build.sh

# Systemlibs for OpenGL
sudo apt update
sudo apt install -y \
  git \
  libegl1 \
  libegl1-mesa \
  libgl1 \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  mesa-utils

sudo ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1  /usr/lib/x86_64-linux-gnu/libGL.so
