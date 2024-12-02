#!/usr/bin/env python3
from . import geom, utils
import yaml

class Config:
    def __init__(self):
        self._default_angular_velocity = geom.radians(30) #(np.pi/5)
        self._default_cartesian_speed = 300/1000.0  #(np.pi/5)
        self._default_joint_speed = geom.radians(30.0)
        self.speed_factor = 1.0
        self.acceleration_time = 0.4
        self.joint_acceleration_time = 1.0
        self.default_grasp_mm = 34.0
        self.verbose = False
        #self.dt = 0.001
        self.dt = 0.01
        self.robot_dt = 0.001

    @property
    def cartesian_speed(self):
        assert(self.speed_factor < 1.5)
        return self._default_cartesian_speed*self.speed_factor

    @property
    def joint_speed(self):
        return self._default_joint_speed*self.speed_factor

    @property
    def angular_velocity(self):
        return self._default_angular_velocity*self.speed_factor

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

def log(*args):
    if cfg.verbose:
        print(*args)
    else:
        pass

def log_error(*args):
    #if cfg.verbose:
    print(*args)
    #else:
    #    pass
