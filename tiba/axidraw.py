import axi
import time
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from . import geom, utils
from . import simplify as simp
from . import matplotlyb as ply
import json
import serial, time
from easydict import EasyDict as edict
reload(geom)

'''
Killing the port
sudo lsof -i tcp:5556
kill -9 {PID}
'''

def print_same_line(s):
    #sys.stdout.write("\r" + str(s))
    print(s, end='\r')
    #sys.stdout.flush()

paper = edict({
    'A4': (11.7, 8.3),
    'A3': (16.5, 11.7),
    'A5': (8.3, 5.8)
})

def cm_to_inch(v):
    return v/2.54

class Plotter:
    def __init__(self,
                 src_rect=None,
                 size=paper.A3,
                 padding=0.7,
                 portrait=False,
                 pen_up_position=60,
                 pen_down_position=0):
        try:
            self.dev = axi.Device()
            self.dev.max_velocity = 0.5
            self.dev.configure()
        except Exception as e:
            print(e)
            self.dev = None

        self.axi_rect = geom.make_rect(0, 0, size[0], size[1]) # inches

        if src_rect is not None:
            if portrait:
                mat = geom.rot_2d(geom.radians(-90))
                src_rect = geom.transform_rect(mat, src_rect)
            else:
                mat = np.eye(3)
            self.aximat = geom.rect_in_rect_transform(src_rect,
                                                      self.axi_rect,
                                                      padding=padding,
                                                      axis=None)@mat
            self.src_rect = src_rect
        else:
            self.aximat = np.eye(3)
            self.src_rect = None
        self.pen_down_position = pen_down_position
        self.pen_up_position = pen_up_position
        self.pen_offset = np.zeros(2)

    def preview(self, S, backend='plt'):
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.aximat, S)
        ply.figure((6, 6))
        ply.rectangle(self.axi_rect, color='r')
        for i, P in enumerate(S):
            ply.stroke(P, 'k') #ply.categorical_color(i))
        ply.setup_2d()
        ply.show(backend)

    def disconnect_axi(self):
        self.dev.close()
        self.dev = None

    def axi_pt(self, p, transform=True):
        if not transform:
            return tuple(p)
        return tuple(geom.affine_transform(self.aximat, p) + self.pen_offset)

    def axi_path(self, P, closed, transform=True):
        Pa = [self.axi_pt(p, transform) for p in P]
        if closed:
            Pa.append(Pa[0])
        if transform:
            Pa = axi.crop_path(Pa, 0. , 0., *tuple(self.axi_rect[1]))[0]
        return Pa

    def home(self):
        if self.dev is None:
            return
        self.dev.home()

    def pen_up(self):
        if self.dev is None:
            return
        if self.pen_down_position > 0:
            self.dev.pen_up_position = self.pen_up_position
            self.dev.configure()
            self.dev.pen_up()
        else:
            self.dev.pen_up()

    def pen_down(self):
        if self.dev is None:
            return
        if self.pen_down_position > 0:
            self.dev.pen_up_position = self.pen_down_position
            print('pendown:', self.dev.pen_up_position)
            self.dev.configure()
            self.dev.pen_up()
        else:
            print('just pd')
            self.dev.pen_down()

    def pen_to(self, where):
        if self.dev is None:
            return

        self.dev.pen_up_position = where
        self.dev.configure()
        self.dev.pen_up()

    def enable_motors(self):
        if self.dev is None:
            return
        self.dev.enable_motors()

    def disable_motors(self):
        if self.dev is None:
            return
        self.dev.disable_motors()


    def plot(self,
             S,
             closed=False,
             max_velocity=3,
             penup_speed=200,
             pen_offset=np.zeros(2),
             penups=True,
             verbose=True,
             transform=True): #200):
        self.pen_offset = pen_offset
        if self.dev is None:
            print("No axidraw device")
            return
        try:
            self.dev.max_velocity = max_velocity
            self.dev.pen_up_speed = penup_speed
            self.dev.pen_down_speed = penup_speed
            self.dev.configure()

            for i, P in enumerate(S):
                if verbose:
                    print_same_line('path %d of %d'%(i, len(S)))
                if penups:
                    self.pen_up()
                self.dev.goto(*self.axi_pt(P[0], transform))
                if penups:
                    self.pen_down()
                self.dev.run_path(self.axi_path(P, closed, transform))
                if penups:
                    self.pen_up()

            self.dev.wait()
        except serial.SerialException:
            print('Axi: Serial error')

    def draw_bounds(self, scale=1):
        Cp = geom.rect_corners(geom.scale_rect(self.src_rect, scale))
        S = []
        S.append(np.array(Cp + [Cp[0]]))
        self.plot(S, max_velocity=3)
        self.home()
        self.close()
