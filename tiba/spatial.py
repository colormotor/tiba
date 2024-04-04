#!/usr/bin/env python3
'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

spatial - Sorting and spatial indexing utilities
'''

import numpy as np
from math import hypot
from collections import defaultdict

# Spatial hashing and sorting adapted from https://github.com/fogleman/axi
class Index(object):
    def __init__(self, points, n=60):

        self.x1 = min(p[0] for p in points)
        self.x2 = max(p[0] for p in points)
        self.y1 = min(p[1] for p in points)
        self.y2 = max(p[1] for p in points)

        # Square increments in indexing,
        # in axi code this is equal, which leads to weird distance results
        dx = abs(self.x2 - self.x1)
        dy = abs(self.y2 - self.y1)
        if dx > dy:
            self.nx = n
            if dx > 0:
                self.ny = max(1, n*(dy/dx))
            else:
                self.ny = 1
        else:
            self.ny = n
            if dy > 0:
                self.nx = max(1, n*(dx/dy))
            else:
                self.nx = 1

        self.bins = defaultdict(list)
        self.size = 0
        for point in points:
            self.insert(point)

    def normalize(self, x, y):
        if abs(self.x2 - self.x1) > 1e-9:
            px = (x - self.x1) / (self.x2 - self.x1)
        else:
            px = 0
        if abs(self.y2 - self.y1) > 1e-9:
            py = (y - self.y1) / (self.y2 - self.y1)
        else:
            py = 0
        i = int(px * self.nx)
        j = int(py * self.ny)

        return (i, j)

    def insert(self, point):
        x, y = point[:2]
        i, j = self.normalize(x, y)
        self.bins[(i, j)].append(point)
        self.size += 1

    def remove(self, point):
        x, y = point[:2]
        i, j = self.normalize(x, y)
        self.bins[(i, j)].remove(point)
        self.size -= 1

    def nearest(self, point):
        x, y = point[:2]
        i, j = self.normalize(x, y)
        points = []
        r = 0
        while not points:
            points.extend(self.ring(i, j, r))
            r += 1
        points.extend(self.ring(i, j, r))
        return min(points,
            key=lambda p: (hypot(x - p[0], y - p[1]), p[1], p[0]))

    def ring(self, i, j, r):
        ''' Fetch a square 'ring' around a point'''
        if r == 0:
            return self.bins[(i, j)]
        result = []

        for p in range(i - r, i + r + 1):
            result.extend(self.bins[(p, j - r)])
            result.extend(self.bins[(p, j + r)])
        for q in range(j - r + 1, j + r):
            result.extend(self.bins[(i - r, q)])

            result.extend(self.bins[(i + r, q)])
        return result

def sort_paths(paths, reversable=True, join_thresh=0):
    if len(paths) < 2:
        return paths

    paths = [[tuple(p) for p in P] for P in paths]
    first = paths[0]
    paths.pop(0)
    result = [first]
    points = []
    print('adding paths')
    for i, path in enumerate(paths):
        x1, y1 = path[0]
        x2, y2 = path[-1]
        points.append((x1, y1, path, False))
        if reversable:
            points.append((x2, y2, path, True))
    print('%d paths'%len(paths))

    index = Index(points)
    print('sorting')
    while index.size > 0:
        x, y, path, reverse = index.nearest(result[-1][-1])
        x1, y1 = path[0]
        x2, y2 = path[-1]
        index.remove((x1, y1, path, False))
        if reversable:
            index.remove((x2, y2, path, True))
        if reverse:
            result.append(list(reversed(path)))
        else:
            result.append(path)

    if join_thresh > 0 and result:
        print('joining')
        joined = [result[0]]
        p = joined[-1][-1]
        for P in result[1:]:
            x, y = P[0]
            if hypot(x - p[0], y - p[1]) < join_thresh:
                joined[-1] += P
            else:
                joined.append(P)
            p = joined[-1][-1]
        result = joined
    return [np.array(P) for P in result]
