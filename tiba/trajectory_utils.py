#!/usr/bin/env python3
import numpy as np
import networkx as nx
from polygonsoup import geom
from itertools import combinations

def tsp_glyph(n, r1=0.5, r2=1.0, center=[0,0]):
    P = geom.shapes.random_radial_polygon(n, r1, r2, center=center)
    I = inverse_path_tsp(P)
    return P[I,:]

def smooth_stroke_lqt(X, only_pos=True, **kwargs):
    import motor.lqt as lqt
    #return st.smooth_stroke(X, ds=1, degree=1)
    dim  = X.shape[1]
    default_params = dict(angle=0,
                          sigma=1.0,
                          order=3,
                          opt_x0=True,
                          stop_motion=True,
                          isotropy=1,
                          activation_sigma=0.5,
                          subd=9,
                          subsample=3)
    default_params.update(kwargs)
    X = lqt.semitied_lqt(X.T, dim=dim, **default_params)
    if only_pos:
        return X[:dim, :].T
    return X.T

def distance(a, b):
    return np.linalg.norm(b[:2] - a[:2])

def path_tsp(P, distfn=distance, **kwargs):
    from networkx.algorithms.approximation import traveling_salesman_problem
    G = nx.Graph()
    n = len(P)
    for i in range(n):
        G.add_node(i)
    # add weighted edges
    for a, b in combinations(range(n), 2):
        G.add_edge(a, b, weight=distfn(P[a], P[b])) #1.0/(1+distance(P[a], P[b])))
    I = traveling_salesman_problem(G, cycle=False, **kwargs)
    return I

def inverse_path_tsp(P, **kwargs):
    def invdist(a, b):
        return 1.0/(1+distance(a, b))
    return path_tsp(P, invdist, **kwargs)
