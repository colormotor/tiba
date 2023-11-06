#!/usr/bin/env python3
from importlib import reload
from roboticstoolbox import Robot
from polygonsoup import plut, geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import roboticstoolbox as rtb
import numbers
import pdb

reload(geom)

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
        plut.stroke(points[start:], 'k', alpha=alpha)
        plut.draw_markers(points[start:], 'k', markersize=1.5, alpha=alpha)
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

    def set_view(self, viewport, trans, rot, fov=60):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        self.mv = geom.trans_3d(trans) @ (geom.rotx_3d(rot[0]) @
                                          geom.roty_3d(rot[1]) @
                                          geom.rotz_3d(rot[2]))
        self.proj = geom.perspective(geom.radians(fov), width/height)


    def polyline(self, P, closed=False):
        self.sketch.canvas.stroke_weight(self._line_width)
        self.sketch.canvas.no_fill()
        self.sketch.canvas.stroke(self.cur_color)
        S = geom.view_3d([P], self.mv, self.proj, viewport=self.viewport)
        for P in S:
            if len(P) > 1:
                self.sketch.canvas.polyline(P, closed=closed)

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
        P = geom.view_3d(P, self.mv, self.proj, viewport=self.viewport, pointcloud=True, clip=True)[0]
        for p in P:
            self.sketch.canvas.circle(p, radius)

    def robot(self, rob):
        T = rob.fkine_all(rob.q)
        P = np.array([x.t for x in T])[1:-6]
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
                  [0, 0, 1, alpha]]
        self.sketch.canvas.stroke_weight(self._line_width)
        for i in range(3):
            self.color(colors[i])
            #self.sketch.canvas.stroke(colors[i])
            self.polyline(segments[i])
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
