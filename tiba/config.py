#!/usr/bin/env python3
from polygonsoup import geom, utils
import yaml

class Config:
    def __init__(self):
        self.default_angular_velocity = geom.radians(30) #(np.pi/5)
        self.default_cartesian_speed = 150/1000.0  #(np.pi/5)
        self.default_joint_speed = 50.0
        self.acceleration_time = 0.5
        #self.dt = 0.001
        self.dt = 0.01
        self.robot_dt = 0.001

    def load(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(data)

cfg = Config()
