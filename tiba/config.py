#!/usr/bin/env python3
from polygonsoup import geom, utils
import yaml

class Config:
    def __init__(self):
        self.default_angular_velocity = geom.radians(30) #(np.pi/5)
        self.default_cartesian_speed = 300/1000.0  #(np.pi/5)
        self.default_joint_speed = 20.0
        self.acceleration_time = 0.4
        self.joint_acceleration_time = 1.0
        self.default_grasp_mm = 34.0
        #self.dt = 0.001
        self.dt = 0.01
        self.robot_dt = 0.001

    def load(self, filename):
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(data)

cfg = Config()


state = lambda: None
state.progress = 0
state.progress_max = 1
state.progress_text = ''
state.progress_active = False
