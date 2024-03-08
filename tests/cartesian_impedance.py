#!/usr/bin/env python3
"""
Uses the cartesian impedance controller to create a sinusoidal
end-effector movement along the robot's y-axis.
"""
import sys

import numpy as np

import panda_py
from panda_py import controllers
from tiba import trajectory_utils as tu

if __name__ == '__main__':
  hostname = '172.16.0.2'
  panda = panda_py.Panda(hostname)
  panda.move_to_start()
  Kp = np.diag([800, 800, 200, 10, 10, 10])
  ctrl = controllers.CartesianImpedance(impedance=Kp, filter_coeff=1.0)
  pos = panda.get_position()
  orn = panda.get_orientation()
  q = panda.q
  runtime = np.pi * 20
  panda.start_controller(ctrl)

  with panda.create_context(frequency=1e3) as ctx:
    while ctx.ok():
      x_d = pos.copy()
      #x_d[1] += 0.1 * np.sin(ctrl.get_time())
      ctrl.set_control(x_d, orn, q)
