#!/usr/bin/env python3
from importlib import reload
from polygonsoup import geom
import tiba.draw as tdraw
import tiba.robot as rob
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import roboticstoolbox as rtb
import pdb

reload(tdraw)
reload(rob)
from tiba.robot import Tiba
from tiba.draw import View3d, SketchView3d
from tiba import planner
reload(planner)

import numpy as np
from collections import namedtuple
import imgui
import swift

params = {'Seed': (5, {'min': 0, 'max': 100}),
          'Gripper mm': (34.0, {'min': 2, 'max': 200}),
          'Gripper state': False,
          'Plane angle': (30, {'min': 0, 'max': 90}),
          'Plane dist' : (0.5, {'min': 0, 'max': 1}),
          'Plane height' : (0.3, {'min': 0, 'max': 1}),
          'Plane offset' : (0.0, {'min': -0.1, 'max': 0.1}),
          'Tsm': {
              'Dist': (4.0, {'min':0, 'max':-10}),
              'Voffset': (0.0, {'min':-1, 'max':1}),
              'rotx': (275, {'min':0, 'max':360}),
              'roty': (0, {'min':0, 'max':360}),
              'rotz': (138, {'min':0, 'max':360}),
          }
}


params = sketch.parameters(params)

class DrawingPad:
    def __init__(self):

        self.frame = np.eye(4)
        self.offset = 0.0

    def update(self, P):
        # Compute frame
        self.points = P
        U, S, mu = principal_components(self.points)
        frame = np.eye(4)
        frame[:3,:3] = U
        frame[:3, 3] = mu
        frame[:3, :3] = orthonormalize_xy(U, np.eye(3))
        self.frame = frame

        # Compute bounding box for sample points
        self.box = compute_frame_aligned_xy_box(self.points, self.frame)

    def get_pen_orientation(self, robot, q):
        z = self.frame[:3,2]
        tcp_frame = robot.pen_tcp_frame(q)
        tcp_frame = align_frame_z(tcp_frame, -z)
        return np.array(UnitQuaternion(SE3(tcp_frame)))

    def fit_trajectory(self, P, padding=100):
        return fit_to_frame_box(P, self.box, self.frame, padding, offset=self.offset)

    def draw(self, v3d):
        v3d.frame(self.frame, 0.5)
        v3d.color(0, 0.4, 1.0, 1.0)
        draw_frame_aligned_xy_box(v3d, self.box, self.frame)
        v3d.color(0, 0.4, 1.0, 0.5)
        draw_frame_aligned_xy_box(v3d, self.box, self.frame, self.offset)
        v3d.color(1, 0, 0, 0.4)
        v3d.points(self.points)

def principal_components(X, resolve_signs=False):
    mu = np.mean(X, axis=0)
    X = (X - mu).T @ (X - mu)
    U, S, Vt = np.linalg.svd(X.T @ X)
    idx = S.argsort()[::-1]
    return U[:,idx], S[idx], mu


def orthonormalize_xy(U, A):
    ''' Compute X and Y axes based on z of U and x, y of A'''
    V = np.array(U)
    V[:3,0] = np.cross(A[:,1], U[:,2], )
    V[:3,1] = np.cross(U[:,2], A[:,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    V[:3,:3] = trnorm(V[:3,:3])
    return V

def align_frame_z(frame, z):
    z = np.array(z)
    frame = np.array(frame)
    frame[:3,2] = z
    frame[:3,0] = np.cross(frame[:3,1], z)
    frame[:3,1] = np.cross(z, frame[:3,0])
    # Must make sure we don't get precision issues
    # https://github.com/bdaiinstitute/spatialmath-python/issues/28
    return trnorm(frame)


def compute_frame_aligned_xy_box(P, frame):
    # Assume we are projecting to XY
    Q = geom.affine_transform(np.linalg.inv(frame), P)
    box = geom.bounding_box(Q[:,:2])
    return box


def get_offset_frame(frame, offset):
    z = frame[:3, 2]
    oframe = np.array(frame)
    oframe[:3, 3] = frame[:3, 3] + z*offset
    return oframe

def draw_frame_aligned_xy_box(v3d, box, frame, offset=0):
    P = geom.close(np.array(geom.rect_corners(box)))
    P = np.hstack([P, np.zeros([P.shape[0], 1])])
    frame = get_offset_frame(frame, offset)
    P = geom.affine_transform(frame, P)
    v3d.polyline(P)


def fit_to_frame_box(P, box, frame, input_padding=10, flip_x=False, flip_y=False, offset=0):
    ''' Fit a polyline to a box projected to XY plane defined by a given frame'''
    P = np.array(P)
    if flip_y:
        P[:,1] = -P[:,1]
    if flip_x:
        P[:,0] = -P[:,0]
    mat = geom.rect_in_rect_transform(geom.bounding_box(P, input_padding), box)
    P = geom.affine_transform(mat, P)

    P = np.hstack([P, np.zeros([P.shape[0], 1])])
    frame = get_offset_frame(frame, offset)
    return geom.affine_transform(frame, P)


v3d = SketchView3d(sketch)
robot = Tiba(52.0, 12.5)
robot.q = robot.qr
pad = DrawingPad()

state = lambda: None
state.Qs = []
state.trajectory = []
state.dt = 0.01


def setup():
    sketch.set_gui_callback(gui)
    create_canvas(700, 600, 300)
    set_color_scale(1.0)
    print(robot.links)

def gui():
    if imgui.button('Launch Swift'):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(robot)
    if imgui.button('Plan trajectory'):
        cur_pos = robot.pen_tcp_pos(robot.q)
        q0 = robot.q
        trajectory = [np.concatenate([pos, state.orientation]) for pos in state.trajectory]
        trajectory = planner.linear_approach(cur_pos, trajectory[0]) + trajectory
        trajectory += planner.linear_approach(trajectory[-1], cur_pos)

        state.Qs, state.dQs = planner.plan_trajectory(robot.urdf_file,
                                                      q0,
                                                      trajectory,
                                                      dt=state.dt)

        # state.Qs, state.dQs = planner.plan_trajectory_constant_orientation(robot.urdf_file,
        #                                                         cur_pose,
        #                                                         state.trajectory,
        #                                                         state.orientation,
        #                                                         14,
        #                                                         state.dt,
        #                                                         q0,
        #                                                         penalty=1e-6)

    if state.Qs:
        if imgui.button('Execute trajectory'):
            env = swift.Swift()
            env.launch(realtime=True)
            env.add(robot)

            for q in state.Qs: #dQs:
                #print(dq)
                robot.q = q
                env.step(state.dt)


def move_to_pose(robot, pose, steps):
    q0 = robot.q
    # Qs = [q0]
    # for t in np.linspace(0, 1, steps):
    #     dest = interpolate_frames(q0,
    sol = robot.ikine_LM(pose, end='tiba_pen_tcp', seed=params.seed)
    # print(sol)
    q = sol.q
    return [q0 + (q - q0)*t for t in np.linspace(0, 1, steps)]

def draw_steps(robot, v3d, Qs, max_steps=200):
    n = min(len(Qs), max_steps)
    I = np.linspace(0, len(Qs)-1, n, dtype=int)
    v3d.color(1, 0.25)
    v3d.line_width(1.0)
    pts = []
    for i in I:
        q = Qs[i]
        P, penP = robot.robot_polylines(q)
        v3d.polyline(P)
        v3d.polyline(penP)
        pts.append(robot.pen_tcp_pos(q)[:3])

    v3d.color(1, 0, 0, 0.4)
    v3d.line_width(4)
    v3d.polyline(pts)

def interpolate_poses(p1, p2, t):
    return SE3(p1 + (p2 - p1)*t)

def linear_trajectory(P, ds):
    Q = [P[0]]
    for a, b in zip(P, P[1:]):
        nsteps = max(2, int(np.linalg.norm(a-b)/ds))
        for t in np.linspace(0, 1, nsteps)[1:]:
            Q.append(a + (b - a)*t)
    return np.array(Q)

def draw():
    np.random.seed(params.seed)
    background(0)
    viewport = geom.make_rect(0, 0, width, height)
    v3d.set_view(viewport, [0.0, params.tsm.voffset, params.tsm.dist],
                 geom.radians(np.array([params.tsm.rotx, params.tsm.roty, params.tsm.rotz])))

    # no_fill()
    # stroke(1, 0.2)
    # stroke_weight(2.0)
    v3d.color(1, 0.2)
    v3d.line_width(2.0)
    v3d.xy_grid(2, 21)

    v3d.frame(np.eye(4), alpha=0.4, scale=0.5)

    q = robot.qr
    P, penP = robot.robot_polylines(q)
    v3d.color(0, 0.5, 1.0)
    v3d.line_width(6)
    v3d.polyline(P)

    penframe = robot.pen_frame(q)
    v3d.frame(penframe, alpha=0.9, scale=0.05)
    v3d.frame(robot.pen_tcp_frame(q))

    v3d.line_width(1)
    v3d.color(1)
    v3d.polyline(penP)

    v3d.text([2,0,0], "2m")

    P = np.random.uniform(-0.4, 0.4, (100, 3))*[1, 1, 0.01]
    P = geom.affine_transform(geom.roty_3d(-geom.radians(params.plane_angle)), P)
    P = P+[params.plane_dist, 0, params.plane_height]
    #v3d.color(1, 0, 1, 0.5)
    #v3d.points(P, 2)
    pad.offset = params.plane_offset
    pad.update(P)
    pad.draw(v3d)
    state.orientation = pad.get_pen_orientation(robot, q)

    no_fill()
    stroke(1)
    Q = np.random.uniform(-100, 100, (17, 2)) + 100
    trajectory = linear_trajectory(Q, 1)
    # trajectory = []
    # r = 100
    # for t in np.linspace(0, np.pi*2, 420):
    #     trajectory.append([np.cos(t)*r + r, np.sin(t)*r + r])

    polyline(trajectory)

    state.trajectory = pad.fit_trajectory(trajectory, padding=130)

    v3d.color(1.0)
    v3d.polyline(state.trajectory)

    #dest = interpolate_poses(SE3(old_frame), SE3(tcp_frame), 0.5)
    #Qs = move_to_pose(robot, dest, 20)
    if state.Qs:
        draw_steps(robot, v3d, state.Qs)
