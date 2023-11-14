#!/usr/bin/env python3
from importlib import reload
import numpy as np
import networkx as nx
from polygonsoup import geom
from itertools import combinations
from scipy.special import erf
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
from scipy.interpolate import interp1d, splprep, splev
from roboticstoolbox.tools.trajectory import ctraj, trapezoidal_func, trapezoidal
from . import trapezoidal
reload(trapezoidal)

from .config import config as cfg
import numbers
import pdb

def is_number(x):
    return isinstance(x, numbers.Number)

def to_pos_orn(pose):
    if (type(pose) == list or
        (type(pose) == np.ndarray and len(pose.shape) == 1)):
        return pose
        #return pose.t, np.array(UnitQuaternion(pose))
    pose = SE3(pose)
    pose = pose.norm()
    return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])


def to_pose(pose):
    # TODO Add normalization?
    if type(pose) == SE3:
        return pose.norm()
    if type(pose) == np.ndarray and len(pose.shape) == 2:
        return SE3(pose).norm()
    pose = SE3(*pose[:3])*(UnitQuaternion(pose[3:])).SE3()
    return pose.norm()


def append_rest(traj, duration):
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    print('appending %d rest'%n)
    return traj + [np.array(traj[-1])]*n

def insert_rest(traj, duration):
    traj = list(traj)
    n = int(duration/cfg.dt)
    if not n:
        return traj
    if not traj:
        print('Cannot add rest to empty trajectory')
        return traj
    print('inserting %d rest'%n)
    return [np.array(traj[0])]*n + traj

def rest_inout(traj, duration):
    traj = insert_rest(traj, duration)
    traj = append_rest(traj, duration)
    return traj
    #return insert_rest(append_rest(traj, duration), duration)

# Trajectory generators
def xy_circles_mm(pose, radius_mm, cycle_duration, cycles=1, ease_duration=0.5, easing=2):
    pose = to_pos_orn(pose)
    pos, orn = pose[:3], pose[3:]

    duration = cycle_duration*cycles
    radius = radius_mm / 1000
    T = gauss_easeinout(0, np.pi*2*cycles, duration, ease_duration, easing=easing)
    pts = [pos + [np.cos(t)*radius, np.sin(t)*radius, 0] for t in T]
    return approach_inout(pose, [np.concatenate([pos, orn]) for pos in pts])

def offset_frame(frame, d):
    print(frame)
    frame = np.array(frame)
    frame[:3, 3] = frame[:3, 3] + d
    return frame

def approach_inout(x, trajectory, kind='trapezoidal', **args):
    if kind=='linear':
        approach = linear_appoach
    elif kind=='trapezoidal':
        approach = trapezoidal_approach
    else:
        approach = gauss_approach

    return (approach(x, trajectory[0], **args) +
            list(trajectory) +
            approach(trajectory[-1], x, **args))


def duration_from_distance(x0, x1, angular_approach_speed=None,
                                   cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    angdist = abs(pose0.angdist(pose1))
    dist = np.linalg.norm(pose0.t - pose1.t)

    def fitts_id(d, w=1):
        return np.log2(d/w + 1)

    duration = max(angdist/angular_approach_speed, dist/cartesian_approach_speed)
    print('fitts comp', duration, duration*fitts_id(duration))
    return duration * (1+fitts_id(duration)) # max(cfg.min_approach_duration, duration+fitts_id(duration))

def linear_approach(x0, x1, duration=0,
                    angular_approach_speed=None,
                    cartesian_approach_speed=None):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if duration <= 0.0:
        duration = duration_from_distance(pose0, pose1,
                                          angular_approach_speed,
                                          cartesian_approach_speed)

    # angular_approach_step = angular_approach_speed*cfg.dt
    # pos_approach_step = cartesian_approach_speed*cfg.dt
    # if duration==0:
    #     angdist = abs(pose0.angdist(pose1))
    #     dist = np.linalg.norm(pose0.t - pose1.t)
    #     # Compute approach steps based on distance
    #     ang_approach_steps = int(angdist/angular_approach_step)
    #     pos_appro ach_steps = int(dist/pos_approach_step)
    #     n = max(ang_approach_steps, pos_approach_steps)
    # else:
    n = int(duration/cfg.dt)
    # Compute appriach keypoints if necessary
    if n > 0:
        approach = pose0.interp(pose1, n)
        approach = [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in approach]
        return approach
    return []

def trapezoidal_approach(x0, x1,
                         angular_approach_speed=None,
                         cartesian_approach_speed=None,
                         debug=False):


    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    angdist = abs(pose0.angdist(pose1))
    dist = np.linalg.norm(pose0.t - pose1.t)

    if angular_approach_speed is None:
        angular_approach_speed = cfg.default_angular_approach_speed
    if cartesian_approach_speed is None:
        cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if angdist/angular_approach_speed > dist/cartesian_approach_speed:
        v_max = angular_approach_speed # * angdist
        d = angdist
    else:
        v_max = cartesian_approach_speed
        d = dist

    a_max = v_max/cfg.acceleration_time

    dt = cfg.dt
    tr = trapezoidal.trapezoidal_profile([0.0], [d], [0], [0], [v_max], [a_max])
    timesteps = int(tr.time / dt)
    time = np.linspace(0, tr.time, timesteps)
    ts = [tr(t)[0][2]/d for t in time]
    approach = [pose0.interp(pose1, t) for t in ts]
    print('approach with ', len(approach))
    return pose_traj_to_posorn(approach)

def approach(a, b, *args, **kwargs):
    return trapezoidal_approach(a, b, *args, **kwargs)

def pose_traj_to_posorn(trajectory):
    return [np.concatenate([pose.t, np.array(UnitQuaternion(pose))]) for pose in trajectory]

def gauss_approach(x0, x1, duration=-1,
                   angular_approach_speed=None,
                   cartesian_approach_speed=None,
                   easing=5,
                   debug=False):
    pose0 = to_pose(x0)
    pose1 = to_pose(x1)

    # if angular_approach_speed is None:
    #     angular_approach_speed = cfg.default_angular_approach_speed
    # if cartesian_approach_speed is None:
    #     cartesian_approach_speed = cfg.default_cartesian_approach_speed

    # if angular_approach_speed is None:
    #     angular_approach_speed = cfg.default_angular_approach_speed
    # if cartesian_approach_speed is None:
    #     cartesian_approach_speed = cfg.default_cartesian_approach_speed

    if duration <= 0.0:
        duration = duration_from_distance(pose0, pose1,
                                          angular_approach_speed,
                                          cartesian_approach_speed)

    # angular_approach_step = angular_approach_speed*cfg.dt
    # pos_approach_step = cartesian_approach_speed*cfg.dt

    # if duration<=0:
    #     print(pose0, pose1)
    #     dist = np.linalg.norm(pose0.t - pose1.t)
    #     print(dist)
    #     try:
    #         angdist = abs(pose0.angdist(pose1))
    #     except ValueError as e:
    #         print(e)
    #         angdist = 0

    #     if debug:
    #         pdb.set_trace()
    #     # Compute approach steps based on distance
    #     ang_approach_steps = int(angdist/angular_approach_step)
    #     pos_approach_steps = int(dist/pos_approach_step)
    #     n = max(ang_approach_steps, pos_approach_steps)
    # else:

    n = int(duration/cfg.dt)
    if n > 0:
        approach = [pose0.interp(pose1, t) for t in gaussspace(0, 1, n, easing=easing)]
        return pose_traj_to_posorn(approach)
    return []



def tsp_glyph(n, r1=0.5, r2=1.0, center=[0,0]):
    P = geom.shapes.random_radial_polygon(n, r1, r2, center=center)
    I = inverse_path_tsp(P)
    return P[I,:]


def smooth_strokes_lqt(S, only_pos=True, penups=True, penup_variability=0.4, **kwargs):
    trajectories = []
    for P in S:
        if penups:
            variability = [1, penup_variability] + [1 for _ in P[1:-1]] + [penup_variability, 1]
        else:
            variability = []
        trajectory = smooth_stroke_lqt(P, variability=variability, **kwargs)
        trajectories.append(trajectory)
    return trajectories

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
    t = np.linspace(a, b, n-2)
    # Hack for zero velocity at start and end
    t = np.concatenate([[a], t, [b]])
    return gauss_easing(t, easing)


def expspace(a, b, n, easing=2):
    t = np.linspace(0, 1, n)
    t = (t-0.5)
    v = np.exp(-(t**2)*(easing**2)) #/(easing**2))
    v = v/np.sum(v)
    return v


def gauss_easeinout(a, b, duration, ease_duration=0.25, easing=5):
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

def bspline(n, pts, closed=False, alpha=1, degree=2, smooth_k=0, w=None, der=0, duration=1.0):
    pts = np.array(pts)
    if closed:
        pts = np.vstack([pts, pts[0]])
    if w is None:
        w = np.ones(len(pts))
    elif is_number(w):
        w = np.ones(len(pts))*w
        if closed:
            w = np.concatenate([w, [w[0]]])

    dim = pts.shape[1]
    u = np.linspace(0, duration, pts.shape[0])
    print('dt_pre:', u[1] - u[0])
    degree = min(degree, pts.shape[0]-1)
    tck, _ = splprep(pts.T, u=u, k=degree, per=closed, s=smooth_k, w=w)

    t = np.linspace(0, duration, n)
    print(t[1] - t[0])
    print('dt_post:', t[1] - t[0])
    if type(der) == list:
        res = []
        for d in der:
            res.append(np.vstack(splev(t, tck, der=d)).T)
        return res

    res = splev(t, tck, der=der)
    return np.vstack(res).T
