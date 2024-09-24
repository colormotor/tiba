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
from . import config
cfg = config.cfg

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


tools = {
    'Default': {
        'tip_dist_mm': 45.0,
        'tip_height_mm': 11.5
    },

    'Probe': {
        'tip_dist_mm': 60.0,
        'tip_height_mm': 0.0,
        'angle': 0.0
    },

    # 'Molotow': {
    #     'tip_dist_mm': 44.0,
    #     'tip_height_mm': 11.0
    # },

    # 'Molotow Thin': {
    #     'tip_dist_mm': 51.0,
    #     'tip_height_mm': 10.0
    # },

    # 'Molotow New': {
    #     'tip_dist_mm': 50.0,
    #     'tip_height_mm': 9.0
    # },

    'Edding': {
        'tip_dist_mm': 50.0,
        'tip_height_mm': 11.0,
        'pen_offset_base_mm': 5.5
    },

    'Calli-brush-01': {
        'tip_dist_mm': 72.0,
        'tip_height_mm': 5.0,
        'pen_offset_base_mm': 9.2,
        'min_radius_mm': 0.5,
        'max_radius_mm': 3.0

    },

    'Squeezer': {
        'is_squeezer': True
    }
}

# Braccio joint limits
q_min_braccio = [0.0, 0.262, 0.0, 0.0, 0.0]
q_max_braccio = [np.pi, 2.879, np.pi, np.pi, np.pi]
braccio_limits = [q_min_braccio, q_max_braccio]

# Franka emika joint limits
q_min_panda = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
q_max_panda = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

q_min_fr3 = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
q_max_fr3 = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

panda_limits = [q_min_panda, q_max_panda]
fr3_limits = [q_min_fr3, q_max_fr3]

class TibaRobot(Robot):
    """
    Robot description and state for Tiba robot (Franka Emika Research 3).
    Assumes the gripper holds a pen holder (see docs) with the pen
    attached at a default angle of 15 degrees,

    ----
    """

    def __init__(self, tip_dist_mm=52.0, tip_height_mm=12.5, angle=15.0,
                 pressure_range=0.0,
                 is_squeezer=False,
                 limits=fr3_limits,
                 **kwargs):
        path = os.path.join(os.path.dirname(__file__), '../urdf/franka_description/')
        if is_squeezer:
            self.xacro_file = os.path.join(path, 'tiba_squeeze.urdf.xacro')
            mappings = {'angle': 45.0}
        else:
            self.xacro_file = os.path.join(path, 'tiba.urdf.xacro')
            mappings={'tip_dist_mm': tip_dist_mm,
                      'tip_height_mm': tip_height_mm,
                      'angle': angle}

        self.urdf_file = os.path.join(path, 'tiba_test.urdf')
        xacro_to_urdf(self.xacro_file, self.urdf_file,
                      tld=path,
                        mappings=mappings)
        links, name, urdf_string, urdf_filepath = self.URDF_read(self.urdf_file,
                                                                 tld=path)
        # print(links)
        super().__init__(
            links,
            name=name,
            manufacturer="Franka Emika",
            gripper_links=links[10],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )


        self.q_min = np.array(limits[0])
        self.q_max = np.array(limits[1])
        self.qd_lim = np.array(
            [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
        )
        self.qdd_lim = np.array([10.0]*7)

        self.qr = np.array([-0.07348355578820649,
                            -0.6406848793249571,
                            0.14414278314814566,
                            -2.2053375955748025,
                            -0.002115707922495155,
                            2.265591827123818,
                            0.8315813678712944])
        self.home = np.array([
                    -0.07348355578820649,
                    -0.6406848793249571,
                    0.14414278314814566,
                    -2.2053375955748025,
                    -0.002115707922495155,
                    2.265591827123818,
                    0.8315813678712944
                ])
        #self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

        self.pressure_range = pressure_range

    def check_joint_trajectory(self, Qs, dQs):
        qmin = np.min(Qs, axis=0)
        qmax = np.max(Qs, axis=0)
        if np.any(qmin < self.q_min) or np.any(qmax > self.q_max):
            print("Joint limits exceeded")
            return False
        dqmax = np.max(np.abs(dQs), axis=0)
        if np.any(dqmax > self.qd_lim):
            print("Joint velocity limits exceeded")
            return False
        ddQ = np.diff(dQs, axis=0)/cfg.robot_dt
        ddqmax = np.max(np.abs(ddQ), axis=0)
        if np.any(ddqmax > self.qdd_lim):
            print("Joint acceleration limits exceeded")
            return False
        return True

    def check_cartesian_trajectories(self, trajectories):
        from . import trajectory_utils as tu
        if not trajectories:
            return
        trajectories = [np.array(P) for P in trajectories]
        xyz_velocities = np.vstack([np.diff(P[:,:3], axis=0) / cfg.dt for P in trajectories])
        xyz_accelerations = np.vstack([np.diff(P[:,:3], n=2, axis=0) / (cfg.dt**2) for P in trajectories])

        max_accel = cfg.cartesian_speed / cfg.acceleration_time
        vmax = np.max(np.abs(xyz_velocities), axis=0)
        amax = np.max(np.abs(xyz_accelerations), axis=0)
        if np.any(vmax > cfg.cartesian_speed):
            print("Cartesian velocity limits exceeded")
            return False
        if np.any(amax > max_accel):
            print("Cartesian acceleration limits exceeded")
            return False

        #if len(trajectories[0][0]) > 3:
        #    # Check angular velocities
        #    poses = [[tu.to_pose(x) for x in P] for P in trajectories if len(P) > 1]
        #    ang_velocities = sum([[tu.angdistance(a,b)/cfg.dt for a, b in zip(P, P[1:])] for P in poses], [])
        #    vmax = np.max(np.abs(ang_velocities))
        #    if vmax > cfg.angular_velocity:
        #        print("Angular velocity limits exceeded")
        #        return False
        return True

    def jacobian(self, q, name):
        if q is None:
            q = self.q
        return self.jacobe(q, start='base', end=name)

    def get_pose(self, q, name):
        if q is None:
            q = self.q
        pose = self.fkine(q, start='base', end=name)
        return pose

    def get_pos(self, q, name):
        if q is None:
            q = self.q
        pose = self.fkine(q, start='base', end=name)
        return np.concatenate([pose.t, np.array(UnitQuaternion(pose))])

    def get_frame(self, q, name):
        return np.array(self.get_pose(q, name))

    def hand_tcp_pose(self, q=None):
        if q is None:
            q = self.q
        return self.get_pose(q, 'tiba_hand_tcp')

    def hand_tcp_pos(self, q=None):
        if q is None:
            q = self.q
        return self.get_pos(q, 'tiba_hand_tcp')

    def pen_tcp_pose(self, q=None):
        if q is None:
            q = self.q
        return self.get_pose(q, 'tiba_pen_tcp')

    def pen_tcp_jacobian(self, q=None):
        if q is None:
            q = self.q
        return self.jacobian(q, 'tiba_pen_tcp')

    def pen_tcp_pos(self, q=None):
        if q is None:
            q = self.q
        return self.get_pos(q, 'tiba_pen_tcp')

    def wrist_frame(self, q=None):
        if q is None:
            q = self.q
        frame = self.fkine(q, start='base', end='tiba_link8')
        return np.array(frame)

    def hand_frame(self, q=None):
        if q is None:
            q = self.q
        frame = self.fkine(q, start='base', end='tiba_hand_tcp')
        return np.array(frame)

    def pen_holder_frame(self, q=None):
        if q is None:
            q = self.q
        frame = self.fkine(q, start='base', end='tiba_holder')
        return np.array(frame)

    def pen_frame(self, q=None):
        if q is None:
            q = self.q
        frame = self.fkine(q, start='base', end='tiba_pen')
        return np.array(frame)

    def pen_tcp_frame(self, q=None):
        if q is None:
            q = self.q
        frame = self.fkine(q, start='base', end='tiba_pen_tcp')
        return np.array(frame)

    def pen_tcp_frame_to_hand(self, frame, q=None):
        if q is None:
            q = self.q
        hand_to_pen = self.fkine(q, start='tiba_holder', end='tiba_pen_tcp')
        frame = frame @ np.linalg.inv(hand_to_pen)
        return frame

    def hand_frame_to_pen_tcp(self, frame, q=None):
        if q is None:
            q = self.q
        hand_to_pen = self.fkine(q, start='tiba_holder', end='tiba_pen_tcp')
        frame = frame * hand_to_pen #np.linalg.inv(hand_to_pen)
        return frame


    def robot_polylines(self, q=None):
        if q is None:
            q = self.q
        T = self.fkine_all(q)[3:11]
        P = [A.t for A in T]
        P += [self.wrist_frame(q)[:3,3]]
        P += [self.pen_holder_frame(q)[:3,3]]
        penP = [self.pen_frame(q)[:3,3], self.pen_tcp_frame(q)[:3,3]]

        return P, penP
