#!/usr/bin/env python3

import spatialmath as sm
import spatialgeometry as sg
import numpy as np
import swift
import roboticstoolbox as rtb
from roboticstoolbox.robot.Robot import Robot
from spatialmath import SE3
from roboticstoolbox.tools import xacro
from spatialmath import SE3, SO3, UnitQuaternion

import os

def xacro_to_urdf(xacro_file, urdf_file, tld=None, mappings={}):
    opts = {
        'output': None,
        'just_deps': False,
        'in_order': True,
        'verbosity': 1,
        'just_includes': False,
        'mappings': mappings,
    }

    xacro.tld = tld

    print("Saving URDF to: ", urdf_file)
    try:
        # open and process file
        doc = xacro.process_file(xacro_file, **opts)
        # open the output file
        out = xacro.open_output(urdf_file)
    except Exception as e:
        print(e)
        return

    out.write(doc.toprettyxml(indent='  ', **xacro.encoding))
    out.close()


class Tiba(Robot):
    """
    Class that imports a Arduino Braccio URDF model
    """

    def __init__(self, tip_dist_mm=52.0, tip_height_mm=12.5, angle=15.0):
        path = os.path.join(os.path.dirname(__file__), '../urdf/franka_description/')
        self.xacro_file = os.path.join(path, 'tiba.urdf.xacro')
        self.urdf_file = os.path.join(path, 'tiba_test.urdf')
        xacro_to_urdf(self.xacro_file, self.urdf_file,
                      tld=path,
                      mappings={'tip_dist_mm': tip_dist_mm,
                                'tip_height_mm': tip_height_mm,
                                'angle': angle})

        links, name, urdf_string, urdf_filepath = self.URDF_read(self.urdf_file,
                                                                 tld=path)
        print(links)
        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[10],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.dt = 0.001



        #self.grippers[0].tool = SE3(0, 0, 0.1034)

        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def get_pose(self, q, name):
        pose = self.fkine(q, start='base', end=name)
        return pose

    def get_pos(self, q, name):
        pose = self.fkine(q, start='base', end=name)
        return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])

    def get_frame(self, q, name):
        return np.array(self.get_pose(q, name))

    def pen_tcp_pose(self, q):
        return self.get_pose(q, 'tiba_pen_tcp')

    def pen_tcp_pos(self, q):
        return self.get_pos(q, 'tiba_pen_tcp')

    def wrist_frame(self, q):
        frame = self.fkine(q, start='base', end='tiba_link8')
        return np.array(frame)

    def hand_frame(self, q):
        frame = self.fkine(q, start='base', end='tiba_hand_tcp')
        return np.array(frame)

    def pen_holder_frame(self, q):
        frame = self.fkine(q, start='base', end='tiba_holder')
        return np.array(frame)

    def pen_frame(self, q):
        frame = self.fkine(q, start='base', end='tiba_pen')
        return np.array(frame)

    def pen_tcp_frame(self, q):
        frame = self.fkine(q, start='base', end='tiba_pen_tcp')
        return np.array(frame)

    def pen_tcp_frame_to_hand(self, frame):
        hand_to_pen = self.fkine(self.qr, start='tiba_holder', end='tiba_pen_tcp')
        frame = frame @ np.linalg.inv(hand_to_pen)
        return frame

    def robot_polylines(self, q):
        T = self.fkine_all(q)[3:11]
        P = [A.t for A in T]
        P += [self.wrist_frame(q)[:3,3]]
        P += [self.pen_holder_frame(q)[:3,3]]
        penP = [self.pen_frame(q)[:3,3], self.pen_tcp_frame(q)[:3,3]]

        return P, penP
