#!/usr/bin/env python3
import numpy as np
import networkx as nx
from polygonsoup import geom
from itertools import combinations
from scipy.special import erf
from spatialmath import SE3, SO3, UnitQuaternion
from .config import config as cfg

def to_pos_orn(pose):
    if (type(pose) == list or
        (type(pose) == np.ndarray and len(pose.shape) == 1)):
        return pose
        #return pose.t, np.array(UnitQuaternion(pose))
    pose = SE3(pose)
    return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])


def to_pose(pose):
    # TODO Add normalization?
    if type(pose) == SE3:
        return pose
    if type(pose) == np.ndarray and len(pose.shape) == 2:
        return SE3(pose)
    return SE3(*pose[:3])*UnitQuaternion(pose[3:]).SE3()


def append_rest(traj, duration):
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    return traj + [np.array(traj[-1])]*n

def insert_rest(traj, duration):
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    return [np.array(traj[0])]*n + traj


# Trajectory generators
def xy_circles_mm(pose, radius_mm, cycle_duration, cycles=1, ease_duration=0.5, easing=2):
    pose = to_pos_orn(pose)
    pos, orn = pose[:3], pose[3:]

    duration = cycle_duration*cycles
    radius = radius_mm / 1000
    T = gauss_easeinout(0, np.pi*2*cycles, duration, ease_duration, easing=easing)
    pts = [pos + [np.cos(t)*radius, np.sin(t)*radius, 0] for t in T]
    return [np.concatenate([pos, orn]) for pos in pts]


def approach_inout(q, trajectory, kind='gauss', **args):
    if kind=='linear':
        approach = linear_appoach
    else:
        approach = gauss_approach

    return (approach(q, trajectory[0], **args) +
            list(trajectory) +
            approach(trajectory[-1], q, **args))


def linear_approach(x0, x1, duration=0,
                    angular_approach_speed=None,
                    cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    angular_approach_step = angular_approach_speed*cfg.dt
    pos_approach_step = cartesian_approach_speed*cfg.dt
    if duration==0:
        angdist = abs(pose0.angdist(pose1))
        dist = np.linalg.norm(pose0.t - pose1.t)
        # Compute approach steps based on distance
        ang_approach_steps = int(angdist/angular_approach_step)
        pos_approach_steps = int(dist/pos_approach_step)
        n = max(ang_approach_steps, pos_approach_steps)
    else:
        n = int(duration/cfg.dt)
    # Compute appriach keypoints if necessary
    if n > 0:
        approach = pose0.interp(pose1, n)
        approach = [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in approach]
        return approach
    return []


def gauss_approach(x0, x1, duration=0,
                   angular_approach_speed=None,
                   cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed
    angular_approach_step = angular_approach_speed*cfg.dt
    pos_approach_step = cartesian_approach_speed*cfg.dt
    if duration==0:
        angdist = abs(pose0.angdist(pose1))
        dist = np.linalg.norm(pose0.t - pose1.t)
        # Compute approach steps based on distance
        ang_approach_steps = int(angdist/angular_approach_step)
        pos_approach_steps = int(dist/pos_approach_step)
        n = max(ang_approach_steps, pos_approach_steps)
    else:
        n = int(duration/cfg.dt)
    if n > 0:
        approach = [pose0.interp(pose1, t) for t in gaussspace(0, 1, n)]
        approach = [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in approach]
        return approach
    return []



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
                          subd=5,
                          subsample=100) # Assuming duration of 0.5
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


def gauss_easing(t, easing=2):
    x = erf((t-0.5)*easing)
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x


def gaussspace(a, b, n, easing=2):
    t = np.linspace(a, b, n)
    return gauss_easing(t, easing)


def expspace(a, b, n, easing=2):
    t = np.linspace(0, 1, n)
    t = (t-0.5)
    v = np.exp(-(t**2)*(easing**2)) #/(easing**2))
    v = v/np.sum(v)
    return v


def gauss_easeinout(a, b, duration, ease_duration, easing=2):
    ''' Returns a gaussian easing function from a to b with given duration and easing_duration'''
    ease_duration = min(duration/2, ease_duration)
    n = int(duration/cfg.dt)
    nease = int(ease_duration/cfg.dt)
    n = max(n, nease*2)
    gauss = expspace(0, 1, nease*2, easing=easing)
    gauss = gauss/np.max(gauss)
    t = [0.0] + list(gauss[:nease]) + list(np.ones(n-nease*2)) + list(gauss[nease:])
    t = np.cumsum(t)
    t = t/np.max(t)
    return a + (b - a)*t
