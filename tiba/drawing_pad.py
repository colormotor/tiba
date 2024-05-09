#!/usr/bin/env python3

'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

drawing_pad - Drawing pad utilities for robot
'''

from importlib import reload
from tqdm import tqdm, trange
import os, sys
import numpy as np
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import matplotlib.pyplot as plt
import pdb

from .config import cfg
from . import trajectory_utils as tu
from . import geom, utils


class DrawingPad:
    ''' Drawing pad abstraction
        It is constructed by specifying points (e.g. by guiding the robot to touch the pad surface),
        and fitting a "box" around the points.
        Calculating the plane is relatively fast so it can be done on the fly when needed.
        This box is used to determine the area over which a drawing will be made.
    '''
    def __init__(self, path='drawing_pad.json'):
        self.path = os.path.expanduser(path)
        self.frame = np.eye(4)
        self.offset = 0.0
        self.box = None
        self.points = []
        self.pos_offset = [0.0, 0.0]
        self.scale = [1.0, 1.0]

    def clear(self):
        self.points = [] #.clear()
        self.save()

    def add_point(self, p):
        p = np.array(p)
        for q in self.points:
            if np.linalg.norm(p - q) < 3e-3:
                print('Redundant point!')
                return
        self.points.append(p)
        self.save()

    def save(self):
        utils.save_json({'points': self.points}, self.path)

    def load(self):
        try:
            pad_data = utils.load_json(self.path)
            self.points = np.array(pad_data['points']) # utils.load_pkl(self.path)
        except FileNotFoundError as e:
            print(e)

    def is_valid(self):
        return len(self.points) > 3 and self.box is not None

    def z(self):
        return self.frame[:3, 2]

    def update(self, P=None):
        # Compute frame
        if P is not None:
            self.points = P

        if len(self.points) < 4:
            return

        U, S, mu = principal_components(self.points)
        frame = np.eye(4)
        frame[:3,:3] = U
        frame[:3, 3] = mu
        frame[:3, :3] = tu.orthonormalize_xy(U[:3,2], np.eye(3))
        self.frame = frame

        # Compute bounding box for sample points
        self.box = compute_frame_aligned_xy_box(self.points, self.frame)
        scaled_box = [np.array(self.box[0]), np.array(self.box[1])]
        size = geom.rect_size(self.box)
        scaled_box[0] += self.pos_offset*size
        scaled_box[1] = scaled_box[0] + self.scale*size

        self.scaled_box = scaled_box


    def align_pose_to_pad(self, pose):
        z = self.frame[:3,2]
        # pose = tu.align_pose_zy(pose, -z, geom.normalize([1.0, 0.0, 0.0]))
        pose = tu.align_frame_z(pose, -z)
        return pose
        #return np.array(UnitQuaternion(SE3(tcp_frame)))

    def get_pen_orientation(self, robot, q):
        z = self.frame[:3,2]
        tcp_pose = robot.pen_tcp_pose(q)
        tcp_pose = self.align_pose_to_pad(tcp_pose)
        return np.array(UnitQuaternion(SE3(tcp_pose)))

        #tcp_frame = robot.pen_tcp_frame(q)
        #tcp_frame = tu.align_frame_z(tcp_frame, -z)
        #return np.array(UnitQuaternion(SE3(tcp_frame)))

    def augment_with_pen_orientation(self, trajectories, robot, q):
        ''' Augment 3d cartesian trajectories with a fixed pen orientation perpendicular to pad
        and encoded as a unit quaternion. The orientation is repeated for each trajectory sample'''
        orn = self.get_pen_orientation(robot, q)
        compound = True
        if not geom.is_compound(trajectories):
            trajectories = [trajectories]
        trajectories = [[np.concatenate([pos, orn]) for pos in trajectory]
                        for trajectory in trajectories]
        if not compound:
            return trajectories[0]
        return trajectories

    def fit_to_box(self, S, padding=0, penup_mm=50.0, get_mat=False):
        ''' Fits an input 2d trajectory to the given 2d box'''
        res = fit_to_box_and_augment(S, self.scaled_box, padding, penup_mm=penup_mm, get_mat=get_mat)
        return res

    def fit_to_frame(self, S, clamp_z=False):
        ''' Fits a 3d trajectory assumed to lie on the XY plane, to the frame of the drawing pad'''
        compound = True
        if not geom.is_compound(S):
            S = [S]
        S = [np.array(P) for P in S]
        if clamp_z:
            for P in S:
                P[:,2] = np.maximum(P[:,2], 0.0)
        S = fit_to_frame(S, self.frame, self.offset)
        if not compound:
            return S[0]
        return S

    def fit_trajectory(self, P, padding=100):
        P = self.fit_to_box(P, padding) # fit_to_box_and_augment(P, self.box, padding)
        return self.fit_to_frame(P)
        #return fit_to_frame_box(P, self.box, self.frame, padding, offset=self.offset)

    def get_box_points(self):
        return get_frame_aligned_xy_box_points(self.box, self.frame, self.offset)
    
    def draw(self, v3d):
        ''' Draw the drawing pad using a View3d or GlView3d object'''
        v3d.frame(self.frame, 0.5)
        if self.is_valid():
            v3d.color(0, 0.4, 1.0, 1.0)
            draw_frame_aligned_xy_box(v3d, self.box, self.frame)
            v3d.color(0, 0.4, 1.0, 0.5)
            draw_frame_aligned_xy_box(v3d, self.box, self.frame, self.offset)
        if len(self.points):
            v3d.color(1, 0, 0, 0.3)
            v3d.points(self.points, 12)
            v3d.color(1, 0, 0, 1)
            v3d.points(self.points, 4)


def principal_components(X, resolve_signs=False):
    mu = np.mean(X, axis=0)
    X = (X - mu).T @ (X - mu)
    U, S, Vt = np.linalg.svd(X.T @ X)
    idx = S.argsort()[::-1]
    return U[:,idx], S[idx], mu


# def orthonormalize_xy(U, A):
#     ''' Compute X and Y axes based on z of U and x, y of A'''
#     V = np.array(U)
#     V[:3,0] = np.cross(A[:,1], U[:,2], )
#     V[:3,1] = np.cross(U[:,2], A[:,0])
#     # Must make sure we don't get precision issues
#     # https://github.com/bdaiinstitute/spatialmath-python/issues/28
#     V[:3,:3] = trnorm(V[:3,:3])
#     return V


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

def get_frame_aligned_xy_box_points(box, frame, offset=0):
    P = geom.close(np.array(geom.rect_corners(box)))
    P = np.hstack([P, np.zeros([P.shape[0], 1])])
    frame = get_offset_frame(frame, offset)
    P = geom.affine_transform(frame, P)
    return P

def draw_frame_aligned_xy_box(v3d, box, frame, offset=0):
    P = get_frame_aligned_xy_box_points(box, frame, offset)
    v3d.polyline(P)

def det22(mat):
    return mat[0,0] * mat[1,1] - mat[0,1]*mat[1,0]

def fit_to_box_and_augment(S, box, input_padding=10, flip_x=False, flip_y=True, penup_mm=0.0, get_mat=False):
    compound = True
    if not geom.is_compound(S):
        S = [S]

    if not S:
        return S
    
    S = [np.array(P) for P in S]
    mat = np.eye(3)

    # Bit messy here
    if flip_y:
        mat = mat @ geom.scaling_2d([1, -1])
        #S = [P*[1,-1] for P in S]
    if flip_x:
        mat = mat @ geom.scaling_2d([-1, 1])
        #S = [P*[-1,1] for P in S]
    mat = geom.rot_2d(geom.radians(-90)) @ mat

    #S = geom.affine_transform(geom.rot_2d(geom.radians(-90)), S)
    mat = mat @ geom.rect_in_rect_transform(geom.bounding_box([P[:,:2] for P in S], input_padding), box)
    scale = np.sqrt(abs(det22(mat)))

    # If we have thickness profile, scale it to fit the drawing
    if S[0].shape[1] > 2:
        Sw = []
        for P in S:
            Q = geom.affine_transform(mat, P[:,:2])
            w = P[:,2]*scale
            Sw.append(np.hstack([Q, w.reshape(-1,1)]))
        S = Sw
    else:
        S = geom.affine_transform(mat, S)

    if penup_mm > 0.0 and S[0].shape[1] < 3:
        S = [np.hstack([P, np.zeros([P.shape[0], 1])]) for P in S]
        up = np.array([0, 0, 1])
        S = [np.vstack([P[0] + up*(penup_mm/1000),
                        P,
                        P[-1] + up*(penup_mm/1000)]) for P in S]


    if not compound:
        res = S[0]
    else:
        res = S
    if get_mat:
        return res, mat
    return res

def fit_to_frame(P, frame, offset=0):
    ''' Fit a polyline to a box projected to XY plane defined by a given frame'''
    frame = get_offset_frame(frame, offset)
    return geom.affine_transform(frame, P)
