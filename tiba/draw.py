#!/usr/bin/env python3
from importlib import reload
from roboticstoolbox import Robot
from . import geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import roboticstoolbox as rtb
import numbers
import pdb

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "browser" #browser"

def is_number(x):
    return isinstance(x, numbers.Number)

def vec(*args):
    return np.array(args)

def esegments_2d(e, q, alpha=0.3, start=0):
    rob = Robot(e)
    print(rob.urdf_string)
    T = rob.fkine_all(q)
    segments = rob.segments()
    # draw all the line segments for the noodle plot

    for i, segment in enumerate(segments):
        linkframes = []
        for link in segment:
            if link is None:
                linkframes.append(rob.base)
            else:
                linkframes.append(T[link.number])
        points = np.array([linkframe.t for linkframe in linkframes])
        #plut.stroke(points[start:], 'k', alpha=alpha)
        #plut.draw_markers(points[start:], 'k', markersize=1.5, alpha=alpha)
        # self.links[i].set_xdata(points[:,0])
        # self.links[i].set_ydata(points[:,1])
    return [p.t for p in T] # T[-1]]

rad = geom.radians

def projected_robot_polygon(rob, q,
                             dist = 4.0,
                             yoffset = -0.2,
                             rotx = 275,
                             roty = 0,
                             rotz = 138,
                             viewport = [[0.0, 0.0], [1.0, 1.0]]):
    viewport = np.array(viewport)
    width, height = geom.rect_size(viewport)
    T = rob.fkine_all(q)
    P = np.array([x.t for x in T])[1:-3]
    mv = geom.trans_3d([0, yoffset, - dist]) @ (geom.rotx_3d(rad(rotx)) @
                                      geom.roty_3d(rad(roty)) @
                                      geom.rotz_3d(rad(rotz)))
    p = geom.perspective(geom.radians(20), width/height)
    S = geom.view_3d(P, mv, p, viewport=viewport)
    return S[0]

class PandaStickFigure:
    def __init__(self):
        self.robot = rtb.models.Panda()

    def polygon(self, q, **kwargs):
        return projected_robot_polygon(self.robot, q, **kwargs)


class View3d:
    def __init__(self):
        pass

    def set_view(self, viewport, trans, rot, fov=60):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        self.mv = geom.trans_3d(trans) @ (geom.rotx_3d(rot[0]) @
                                          geom.roty_3d(rot[1]) @
                                          geom.rotz_3d(rot[2]))
        self.proj = geom.perspective(geom.radians(fov), width/height)

    def polygons(self, P):
        P = geom.view_3d(P, self.mv, self.proj, viewport=self.viewport)
        return P

    def points(self, P):
        P = geom.view_3d(P, self.mv, self.proj, viewport=self.viewport, pointcloud=True, clip=False)
        return P

def get_color(*args):
    if len(args) == 1:
        if is_number(args[0]):
            v = args[0]
            return (v, v, v, 1)
        else:
            c = np.ones(4)
            c[:len(args[0])] = args[0]
            return c
    elif len(args) == 2:
        if is_number(args[0]) and is_number(args[1]):
            v = args[0]
            return (v, v, v, args[1])
        else:
            print('Invalid color')
            raise ValueError
    else:
        c = np.ones(4)
        c[:len(args)] = args
        return c

class SketchView3d:
    def __init__(self, sketch):
        self.sketch = sketch
        self.cur_color = (1.0, 1.0, 1.0, 1.0)
        self._line_width = 1.0
        self.saved_viewport = None
        self.fov = 60

    def set_view(self, viewport, trans, rot, fov=None):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        if fov is not None:
            self.fov = fov
        rot = np.array(rot)
        if len(rot.shape) == 1: #Euler
            rot = (geom.rotx_3d(rot[0]) @
                   geom.roty_3d(rot[1]) @
                   geom.rotz_3d(rot[2]))
        self.proj = geom.perspective(geom.radians(self.fov), width/height)
        self.mv = geom.trans_3d(trans) @ rot

    def canvas(self):
        return self.sketch.canvas

    def save_viewport(self):
        self.saved_viewport = self.viewport

    def set_viewport(self, viewport):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        self.proj = geom.perspective(geom.radians(self.fov), width/height)

    def restore_viewport(self):
        if self.saved_viewport is None:
            print('No saved viewport')
            return
        self.set_viewport(self.saved_viewport)

    def get_polylines(self, S):
        S = geom.view_3d(S, self.mv, self.proj, viewport=self.viewport)
        return S

    def polyline(self, P, closed=False):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        S = geom.view_3d([P], self.mv, self.proj, viewport=self.viewport)
        for P in S:
            if len(P) > 1:
                self.sketch.canvas.polyline(P, closed=closed)

    def arrow(self, a, b):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        P = np.array([a, b])
        S = geom.view_3d([P], self.mv, self.proj, viewport=self.viewport)
        for P in S:
            if len(P) == 2:
                self.sketch.canvas.arrow(*P)

    def line(self, a, b):
        self.polyline([a, b])

    def stroke_shape(self, S, closed=False):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        S = geom.view_3d(S, self.mv, self.proj, viewport=self.viewport)
        S = [P for P in S if len(P) > 1]
        self.sketch.canvas.shape(S, closed)

    def xy_grid(self, size, n, z=0):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        polygons = []
        for x in np.linspace(-size, size, n):
            polygons += [[[x, -size, z], [x, size, z]]]
        for y in np.linspace(-size, size, n):
            polygons += [[[-size, y, z], [size, y, z]]]
        self.stroke_shape(polygons, closed=False)

    def line_width(self, width):
        self._line_width = width

    def color(self, *args):
        self.cur_color = get_color(*args)

    def points(self, P, radius=2):
        self.sketch.canvas.no_stroke()
        self.sketch.canvas.fill(self.cur_color)
        P = geom.view_3d(P, self.mv, self.proj, viewport=self.viewport, pointcloud=True, clip=True)
        if P:
            for p in P[0]:
                self.sketch.canvas.circle(p, radius)

    def robot(self, rob):
        T = rob.fkine_all(rob.q)
        P = np.array([x.t for x in T])[2:-6]
        self.polyline(P)

    def frame(self, mat, alpha=1.0, scale=0.1):
        pos = mat[:3, -1]
        x = mat[:3, 0]*scale
        y = mat[:3, 1]*scale
        z = mat[:3, 2]*scale
        segments = [[pos, pos+x],
                    [pos, pos+y],
                    [pos, pos+z]]
        segments = [np.array(P) for P in segments]

        color = self.cur_color
        colors = [[1, 0, 0, alpha],
                  [0, 1, 0, alpha],
                  [0, 0.3, 1, alpha]]
        self.sketch.canvas.stroke_weight(self._line_width)
        for i in range(3):
            self.color(colors[i])
            self.arrow(*segments[i])
            #self.sketch.canvas.stroke(colors[i])
            #self.polyline(segments[i])
        self.color(color)

    def box(self, bmin, bmax):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        S = []
        S.append(np.array([[min[0], min[1], min[2]],
                           [max[0], min[1], min[2]],
                           [max[0], max[1], min[2]],
                           [min[0], max[1], min[2]]]))
        S.append(np.array([[min[0], min[1], max[2]],
                           [max[0], min[1], max[2]],
                           [max[0], max[1], max[2]],
                           [min[0], max[1], max[2]]]))
        self.stroke_shape(S, closed=True)

    def cuboid(self, center, size):
        center = np.array(center)
        size = np.array(size)

        bmin = center - size/2
        bmax = center + size/2
        self.box(bmin, bmax)

    def text(self, pos, text, center=True):
        self.sketch.canvas.no_stroke()
        self.sketch.canvas.fill(self.cur_color)
        pts = geom.view_3d([pos], self.mv, self.proj, viewport=self.viewport, pointcloud=True, clip=True)
        if pts:
            self.sketch.canvas.text(pts[0][0], text, center=center)

class GlView3d:
    def __init__(self, sketch):
        from tiba.gfx import renderer
        self.cur_color = (1.0, 1.0, 1.0, 1.0)
        self._line_width = 1.0
        self.saved_viewport = None
        self.fov = 60
        self.rend = renderer.Renderer()
        self.sketch_view = SketchView3d(sketch)

    def release(self):
        print("Releasing GlView3d")
        del self.rend

    def set_view(self, viewport, trans, rot, fov=None):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        if fov is not None:
            self.fov = fov
        self.rend.viewport(width, height)
        self.rend.perspective(self.fov, width, height)
        rot = np.array(rot)
        if len(rot.shape) == 1: #Euler
            rot = (geom.rotx_3d(rot[0]) @
                   geom.roty_3d(rot[1]) @
                   geom.rotz_3d(rot[2]))
        self.rend.apply_matrix(geom.trans_3d(trans) @ rot)
        self.sketch_view.set_view(viewport, trans, rot, fov)

    def save_viewport(self):
        self.saved_viewport = self.viewport

    def set_viewport(self, viewport):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        self.proj = geom.perspective(geom.radians(self.fov), width/height)

    def restore_viewport(self):
        if self.saved_viewport is None:
            print('No saved viewport')
            return
        self.set_viewport(self.saved_viewport)

    def polyline(self, P, closed=False):
        P = np.array(P)
        self.rend.stroke(P, closed)

    def get_polylines(self, S):
        return self.sketch_view.get_polylines(S)

    # def arrow(self, a, b):
    #     self.sketch.canvas.stroke_weight(self._line_width)
    #     self.sketch.canvas.no_fill()
    #     self.sketch.canvas.stroke(self.cur_color)
    #     P = np.array([a, b])
    #     S = geom.view_3d([P], self.mv, self.proj, viewport=self.viewport)
    #     for P in S:
    #         if len(P) == 2:
    #             self.sketch.canvas.arrow(*P)

    def line(self, a, b):
        self.polyline([a, b])

    def stroke_shape(self, S, closed=False):
        self.rend.stroke_shape(S, closed)

    def xy_grid(self, size, n, z=0):
        polygons = []
        for x in np.linspace(-size, size, n):
            polygons += [[[x, -size, z], [x, size, z]]]
        for y in np.linspace(-size, size, n):
            polygons += [[[-size, y, z], [size, y, z]]]
        self.stroke_shape(polygons, closed=False)

    def line_width(self, width):
        self._line_width = width
        self.rend.line_width(width)

    def color(self, *args):
        self.cur_color = get_color(*args)
        self.rend.color(self.cur_color)

    def points(self, P, radius=2):
        self.rend.points(np.array(P), radius)

    def get_robot_configuration_polylines(self, rob, Q):
        S = []
        for q in Q:
            T = rob.fkine_all(rob.q)
            P = np.array([x.t for x in T])[2:-6]
            S.append(P)
        return S

    def robot(self, rob):
        T = rob.fkine_all(rob.q)
        P = np.array([x.t for x in T])[2:-6]
        self.polyline(P)

    def frame(self, mat, alpha=1.0, scale=0.1):
        pos = mat[:3, -1]
        x = mat[:3, 0]*scale
        y = mat[:3, 1]*scale
        z = mat[:3, 2]*scale
        segments = [[pos, pos+x],
                    [pos, pos+y],
                    [pos, pos+z]]
        segments = [np.array(P) for P in segments]

        color = self.cur_color
        colors = [[1, 0, 0, alpha],
                  [0, 1, 0, alpha],
                  [0, 0.3, 1, alpha]]
        lw = self._line_width
        self.rend.line_width(self._line_width) #sketch.canvas.stroke_weight(self._line_width)
        for i in range(3):
            self.color(colors[i])
            self.polyline(segments[i])
            #self.arrow(*segments[i])
            #self.sketch.canvas.stroke(colors[i])
            #self.polyline(segments[i])
        self.color(color)
        self.line_width(lw)

    def box(self, bmin, bmax):
        #self.sketch.canvas.stroke_weight(self._line_width)
        #self.sketch.canvas.no_fill()
        #self.sketch.canvas.stroke(self.cur_color)
        S = []
        S.append(np.array([[min[0], min[1], min[2]],
                           [max[0], min[1], min[2]],
                           [max[0], max[1], min[2]],
                           [min[0], max[1], min[2]]]))
        S.append(np.array([[min[0], min[1], max[2]],
                           [max[0], min[1], max[2]],
                           [max[0], max[1], max[2]],
                           [min[0], max[1], max[2]]]))


        self.stroke_shape(S, closed=True)

    def cuboid(self, center, size):
        center = np.array(center)
        size = np.array(size)

        bmin = center - size/2
        bmax = center + size/2
        self.box(bmin, bmax)

    def text(self, pos, text, center=True):
        self.sketch.canvas.no_stroke()
        self.sketch.canvas.fill(self.cur_color)
        pts = geom.view_3d([pos], self.mv, self.proj, viewport=self.viewport, pointcloud=True, clip=True)
        if pts:
            self.sketch.canvas.text(pts[0][0], text, center=center)



def plot_cartesian_trajectory_stats(trajectories, max_speed=None, accel_time=None, backend='plotly', figsize=(10,10)):
    from . import config
    from . import matplotlyb as ply

    if max_speed is None:
        max_speed = config.cfg.cartesian_speed
    if accel_time is None:
        accel_time = config.cfg.acceleration_time

    velocities = [np.diff(P, axis=0) / config.cfg.dt for P in trajectories]
    accelerations = [np.diff(P, n=2, axis=0) / (config.cfg.dt**2) for P in trajectories]
    max_accel = max_speed / accel_time

    ply.figure(figsize=figsize, title='Cartesian trajectory')

    Tv = 0.0
    Ta = 0.0

    for i, X in enumerate(trajectories):
        #speed = velocities[i] #tu.compute_speed(X, cfg.dt)
        #accel = accelerations[i] #tu.compute_accel(X, cfg.dt)
        tv = np.linspace(0, (len(velocities[i])-1)*config.cfg.dt, len(velocities[i])) + Tv
        ta = np.linspace(0, (len(accelerations[i])-1)*config.cfg.dt, len(accelerations[i])) + Ta
        ply.subplot(1, 1, title='Velocity')
        ply.plot(tv, velocities[i][:,0], color=ply.categorical_color(i))
        ply.plot(tv, velocities[i][:,1], color=ply.categorical_color(i))
        ply.plot(tv, velocities[i][:,2], color='b', alpha=0.5)
        ply.subplot(1, 2, title='Acceleration')
        ply.plot(ta, accelerations[i][:,0], color=ply.categorical_color(i))
        ply.plot(ta, accelerations[i][:,1], color=ply.categorical_color(i))
        ply.plot(ta, accelerations[i][:,2], color='b', alpha=0.5)
        Tv = tv[-1] + config.cfg.dt
        Ta = ta[-1] + config.cfg.dt

    ply.subplot(1, 1, title='Velocity')
    ply.rectangle([0, -max_speed], [Tv, max_speed], color='k', alpha=0.1, fill=True)
    ply.subplot(1, 2, title='Acceleration')
    ply.rectangle([0, -max_accel], [Ta, max_accel], color='k', alpha=0.1, fill=True)


    ply.subplot(2, 1, title='Trajectory 2d')
    ply.stroke(trajectories, 'k')
    ply.axis_equal(True)

    ply.subplot(2, 2, title='Trajectory 3d')
    box = geom.bounding_box(trajectories)

    for X in trajectories:
        X = np.array(X)
        vals = (X[:,2])/(box[1][2] - box[0][2])
        ply.stroke3d(X, color=[ply.cmap(z, 'magma') if z >=0 else ply.cmap(-z, 'viridis') for z in vals])
    ply.show(backend)
