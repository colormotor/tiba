#!/usr/bin/env python3
from polygonsoup import geom, utils

class Config:
    def __init__(self):
        self.default_angular_approach_speed = geom.radians(30) #(np.pi/5)
        self.default_cartesian_approach_speed = 150/1000.0  #(np.pi/5)
        self.default_joint_approach_speed = 50.0
        self.acceleration_time = 0.5
        self.min_approach_duration = 0.5
        #self.dt = 0.001
        self.dt = 0.01
        self.robot_dt = 0.001

cfg = Config()
