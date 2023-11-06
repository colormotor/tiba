#!/usr/bin/env python3
import moderngl as mgl
import numpy as np
import numbers
import os
from polygonsoup import geom

def is_number(x):
    return isinstance(x, numbers.Number)

sizes = {'1f': 1*4,
         '2f': 2*4,
         '3f': 3*4,
         '4f': 4*4}

types = {'1f': 'f4',
         '2f': 'f4',
         '3f': 'f4',
         '4f': 'f4'}

class DynamicVAO:
    def __init__(self, prog,
                 attributes,
                 reserve_vertices=1000,
                 reserve_indices=0,
                 index_size=4):
        self.prog = prog
        self.attributes = attributes
        self.index_size = index_size
        self.ibo = None
        self.vao = None
        self.vbos = []
        self.resize(reserve_vertices, reserve_indices)

    def resize(self, reserve_vertices, reserve_indices, update_indices=True):
        self.release(update_indices)

        self.reserved_vertices = reserve_vertices
        self.reserved_indices = reserve_indices
        print('resize')
        self.vbos = []

        for fmt, name in self.attributes:
            size = sizes[fmt]
            print(size)
            print(reserve_vertices)
            buffer_size = reserve_vertices * size
            vbo = self.prog.ctx.buffer(reserve=buffer_size, dynamic=True)
            self.vbos.append(vbo)

        if update_indices:
            if reserve_indices:
                buffer_size = reserve_indices * self.index_size
                ibo = self.prog.ctx.buffer(reserve=buffer_size, dynamic=True)
                self.ibo = ibo
            else:
                self.ibo = None

        self.vao = self.prog.ctx.vertex_array(self.prog,
                                              [(vbo, fmt, name) for vbo, (fmt, name)
                                               in zip(self.vbos, self.attributes)],
                                              index_buffer=self.ibo)


    def release(self, release_indices=True):
        if self.ibo is not None and release_indices:
            self.ibo.release()
        for vbo in self.vbos:
            vbo.release()
        if self.vao is not None:
            self.vao.release()

    def update(self, vertices, indices=None):
        resize = False
        reserve_indices = 0

        reserve_vertices = len(vertices[0])
        if reserve_vertices > self.reserved_vertices:
            resize = True

        update_indices = False
        if indices is not None and len(indices) > self.reserved_indices:
            resize = True
            reserve_indices = len(indices)
            update_indices = True

        if resize:
            self.resize(reserve_vertices, reserve_indices, update_indices)

        if indices is not None:
            self.ibo.write(np.array(indices).astype('i4'))

        for vbo, V, (fmt, name) in zip(self.vbos, vertices, self.attributes):
            vbo.write(V.copy(order='C').astype(types[fmt]))

    def render(self, primitives):
        self.vao.render(primitives)

def local_path(path):
    return os.path.join(os.path.dirname(__file__), path)

def read_local(path):
    try:
        with open(local_path(path), 'r') as f:
            return f.read()
    except FileNotFoundError as e:
        print(e)
        return ''

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

def shader_matrix(m):
    return np.asarray(m.T.astype('f4'), order='C')

class Renderer:
    def __init__(self):
        self.ctx = mgl.create_context()
        self.cur_color = (1.0, 1.0, 1.0, 1.0)
        self.line_width(1.0)

        self.polyline_prog = self.ctx.program(vertex_shader=read_local('shaders/polyline.vert'),
                                              fragment_shader=read_local('shaders/polyline.frag'),
                                              geometry_shader=read_local('shaders/polyline.geom'))

        self.points_prog = self.ctx.program(vertex_shader=read_local('shaders/disk.vert'),
                                              fragment_shader=read_local('shaders/disk.frag'))


        self.polyline_vao = DynamicVAO(self.polyline_prog,
                                       [("3f", "position")],
                                       reserve_vertices=1000,
                                       reserve_indices=1000)
        self.points_vao = DynamicVAO(self.points_prog,
                                       [("3f", "position"),
                                        ("1f", "radius")],
                                       reserve_vertices=1000)
        self.ctx.enable(mgl.BLEND)
        self.matrix_stack = [np.eye(4)]

    def __del__(self):
        print("Release")
        self.polyline_vao.release()
        self.polyline_prog.release()

    def line_width(self, width):
        self.ctx.line_width = width

    def clear(self, *args):
        self.ctx.clear(*get_color(*args))

    def color(self, *args):
        self.cur_color = get_color(*args)

    def viewport(self, width, height):
        self.size = [width, height]
        self.ortho_projection = geom.ortho(-width/2, width/2, -height/2, height/2, 0.5, -0.5)
        self.matrix_stack = [np.eye(4)]

    def perspective(self, fov, width, height, near=0.1, far=1000.0):
        self.mp = geom.perspective(geom.radians(fov), width/height, near, far)
        self.matrix_stack = [np.eye(4)]
        self.mvp = self.mp @ self.matrix_stack[-1]

    def push_matrix(self, m=None):
        if m is None:
            m = np.eye(4)
        self.matrix_stack.append(self.matrix_stack[-1] @ m)
        self.mvp = self.mp @ self.matrix_stack[-1]

    def pop_matrix(self):
        self.matrix_stack.pop()
        self.mvp = self.mp @ self.matrix_stack[-1]

    def apply_matrix(self, m):
        self.matrix_stack[-1] = self.matrix_stack[-1] @ m
        self.mvp = self.mp @ self.matrix_stack[-1]

    def stroke(self, P, closed=False):
        I = np.arange(len(P) + 2) - 1
        I[0], I[-1] = 0, len(P) - 1

        if closed:
            if np.allclose(P[0], P[1]):
                I = np.arange(len(P) + 2) - 1
                I[0], I[-1] = 0, len(P) - 1
            else:
                I = np.arange(len(P) + 3) - 1
                I[0], I[-2], I[-1] = len(P) - 1, 0, 1
        else:
            I = np.arange(len(P) + 2) - 1
            I[0], I[-1] = 0, len(P) - 1

        self.polyline_vao.update([P], I)

        prog = self.polyline_prog

        prog["linewidth"].value = self.ctx.line_width
        prog["viewport"].value = self.size
        prog["miter_limit"].value = -1.0
        prog['color'].value = self.cur_color
        prog["antialias"].value = 1.4
        prog["projection"].write(shader_matrix(self.ortho_projection))
        prog["mvp"].write(shader_matrix(self.mvp))

        self.polyline_vao.render(mgl.LINE_STRIP_ADJACENCY)

    def points(self, P, radius):
        self.points_vao.update([P, np.ones(len(P))*radius])
        prog = self.points_prog
        prog['color'].value = self.cur_color
        prog["viewport"].value = self.size
        prog["ortho"].write(shader_matrix(self.ortho_projection))
        prog["mvp"].write(shader_matrix(self.mvp))
        self.points_vao.render(mgl.POINTS)
