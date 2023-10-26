#!/usr/bin/env python3
import moderngl as mgl
import numpy as np
import numbers

def is_number(x):
    return isinstance(x, numbers.Number)

sizes = {'1f': 1*4,
         '2f': 2*4,
         '3f': 3*4,
         '4f': 4*4}

class DynamicVAO:
    def __init__(self, prog, attributes, reserve_vertices=1000, reserve_indices=0, index_size=4):
        self.prog = prog
        self.attributes = attributes

        self.vbos = []

        for fmt, name in attributes:
            size = sizes[fmt]
            self.vbos.append(prog.ctx.buffer(reserve_vertices * size))
        if reserve_indices:
            self.ibo = prog.ctx.buffer(reserve_indices * index_size)
        else:
            self.ibo = None

        self.vao = prog.ctx.vertex_array(prog,
                                         [(vbo, fmt, name) for vbo, (fmt, name)
                                          in zip(self.vbos, self.attributes)], index_buffer=self.ibo)

    def resize(self, reserve_vertices, reserve_indices):
        self.reserved_vertices = reserve_vertices
        self.reserved_indices = reserve_indices

    def release(self):
        if self.ibo is not None:
            self.ibo.release()
        for vbo in self.vbos:
            vbo.release()
        self.vao.release()

    def update(self, vertices, indices=None):
        resize = False
        reserve_indices = 0
        if indices is not None and len(indices) > self.reserved_indices:
            resize = True
            reserve_indices = len(indices)

        reserve_vertices = len(vertices[0])
        if reserve_vertices > self.reserved_vertices:
            resize = True

        if resize:
            self.release()
            self.resize(reserve_vertices, reserve_indices)

        if indices is not None:
            self.ibo.write(np.array(indices).astype('i4'))

        for vbo, V, (fmt, name) in zip(self.vbos, vertices, self.attributes):
            vbo.write(V.copy(order='C').astype(fmt))

    def render(self, primitives):
        self.vao.render(primitives)


class Renderer:
    def __init__(self):
        self.ctx = mgl.create_context()
        self.cur_color = (1.0, 1.0, 1.0, 1.0)

    def line_width(self, width):
        self.ctx.line_width = width

    def color(self, *args):
        if len(args) == 1:
            if is_number(args[0]):
                v = args[0]
                self.cur_color = (v, v, v, 1)
            else:
                c = np.ones(4)
                c[:len(args[0])] = args[0]
                self.cur_color = c
        elif len(args) == 2:
            if is_number(args[0]) and is_number(args[1]):
                v = args[0]
                self.cur_color = (v, v, v, args[1])
            else:
                print('Invalid color')
                raise ValueError
        else:
            c = np.ones(4)
            c[:len(args)] = args
            self.cur_color = c

    def stroke(P, closed=False):
