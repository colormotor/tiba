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

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "browser" #browser"


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
        self.saved_viewport = None
        self.fov = 60

    def set_view(self, viewport, trans, rot, fov=None):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        if fov is not None:
            self.fov = fov
        self.proj = geom.perspective(geom.radians(self.fov), width/height)
        self.mv = geom.trans_3d(trans) @ (geom.rotx_3d(rot[0]) @
                                          geom.roty_3d(rot[1]) @
                                          geom.rotz_3d(rot[2]))

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

    def set_view(self, viewport, trans, rot, fov=None):
        self.viewport = [np.array(viewport[0]),
                         np.array(viewport[1])]
        width, height = geom.rect_size(self.viewport)
        if fov is not None:
            self.fov = fov
        self.rend.viewport(width, height)
        self.rend.perspective(self.fov, width, height)
        self.rend.apply_matrix(geom.trans_3d(trans) @ (geom.rotx_3d(rot[0]) @
                                          geom.roty_3d(rot[1]) @
                                          geom.rotz_3d(rot[2])))
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

def plotly_plot(entries, title='', legend_names=[]):
    def categorical(i, alpha=1.0):
        return 'rgba(%.3f, %.3f, %.3f, %.3f)'%tuple([v*255 for v in plut.default_color(i)] + [alpha*255])

    def plotly_color(clr):
        if type(clr) is str:
            return clr
        if len(clr)==3:
            return 'rgb(%.3f, %.3f, %.3f)'%tuple([v*255 for v in clr])
        elif len(clr)==4:
            return 'rgba(%.3f, %.3f, %.3f, %.3f)'%tuple([v*255 for v in clr])

    if type(entries[0]) is not list:
        entries = [entries]
    rows = len(entries)
    cols = int(np.max([len(e) for e in entries]))
    print('Cols', cols)
    specs = []
    titles = []

    for i in range(rows):
        specs.append([])
        for j in range(cols):

            spec = {}
            entry = entries[i][j]

            if entry is None or not entry:
                #titles.append('')
                specs[-1].append(None)
            else:
                if 'type' in entry:
                    tp = entry['type']
                    if tp == 'limits':
                        tp = 'bar'
                    spec['type'] = tp #entry['type']
                if 'colspan' in entry:
                    spec['colspan'] = entry['colspan']
                    if 'title' in entry:
                        titles.append(entry['title'])
                    else:
                        titles.append('')
                specs[-1].append(spec)

    print(specs)
    print('Cols', cols)
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles)
    firstlegend = True
    for i in range(rows):
        for j in range(cols):
            spec = {}
            entry = entries[i][j]
            if not entry or entry is None:
                continue
            if 'type' not in entry or entry['type'] == 'scatter':
                values = entry['values']
                for k, value in enumerate(values):
                    if k < len(legend_names):
                        name = legend_names[k]
                    else:
                        name = None
                    value = np.array(value)
                    print('nnn', len(value.shape))
                    if len(value.shape) == 1:
                        fig.add_trace(go.Scatter(y=value, marker_color=categorical(k),
                                                 legendgroup = '1',
                                                 showlegend=firstlegend,
                                                 name=name),
                                      row=i+1, col=j+1)
                    else:
                        fig.add_trace(go.Scatter(x=value[0], y=value[1],
                                                 marker_color=categorical(k),
                                                 legendgroup = '1',
                                                 showlegend=firstlegend,
                                                 name=name),
                                      row=i+1, col=j+1)
                firstlegend = False
            elif entry['type'] == 'surface':
                values = entry['values']
                if 'colors' in entry:
                    colors = entry['colors']
                else:
                    colors = [None for k in range(len(values))]
                if 'marker_size' in entry:
                    marker_sizes = np.ones(len(values))*entry['marker_size']
                else:
                    marker_sizes = [0 for k in range(len(values))]
                if 'line_width' in entry:
                    line_width = np.ones(len(values))*entry['line_width']
                else:
                    line_width = np.ones(len(values))*1.0

                for k, (color, X) in enumerate(zip(colors, values)):
                    if color is None:
                        color = categorical(k)
                    else:
                        color = plotly_color(color)

                    if k < len(legend_names):
                        name = legend_names[k]
                    else:
                        name = None
                    X = np.array(X)

                    marker_size = marker_sizes[k]
                    if True: #marker_size > 0:
                        marker = dict(
                            size=marker_size,
                            color=color, #categorical(k), #X[:,2],
                            #colorscale='Viridis',
                        )
                    else:
                        marker = None

                    fig.add_trace(go.Scatter3d(
                        x=X[:,0], y=X[:,1], z=X[:,2],
                        marker=marker,
                        line=dict(
                            color=color,
                            width=line_width[k]
                        ),
                        legendgroup = '1',
                        showlegend=firstlegend,
                        name=name),
                        row=i+1, col=j+1)
                firstlegend = False
            elif entry['type'] == 'limits': # Plot joint limits
                X = entry['X']
                min_ref = np.array(entry['min'])
                n = len(min_ref)
                max_ref = np.array(entry['max'])
                min = np.array([np.min(X[:,i]) for i in range(n)])
                max = np.array([np.max(X[:,i]) for i in range(n)])
                colors = [categorical(i) for i in range(n)]
                limit_colors = [categorical(i, 0.5) for i in range(n)]
                x = np.linspace(0, n-1, n)
                fig.add_trace(go.Bar(x=x, y=max_ref-min_ref, base=min_ref, marker_color=limit_colors),
                                      row=i+1, col=j+1)
                fig.add_trace(go.Bar(x=x, y=max-min, base=min, marker_color=colors, showlegend=False),
                                      row=i+1, col=j+1)
                firstlegend = False

    fig.layout.update(title=title, showlegend=True, barmode='relative')
    fig.show()
