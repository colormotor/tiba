#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import numpy as np
import importlib
panda_loader = importlib.find_loader('panda_py')
if panda_loader is not None:
    import panda_py
    from panda_py import libfranka
    from panda_py import controllers
    has_robot = True
else:
    has_robot = False

from tiba.interface import FrankaButton, TibaInterface
from tiba.robot import TibaRobot
from tiba.draw import View3d, SketchView3d, GlView3d
from tiba import planner, cfg


if has_robot:
    hostname = '172.16.0.2'
    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)
else:
    gripper = None
    desk = None
    panda = None

robot = TibaRobot(45, 11.5) # Molotow
itiba = TibaInterface(robot, panda, gripper)
robot.q = robot.qr

def msg_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Msg: %s", data.data)
    if 'exec:' in data.data:
        cmd = data.data.split(':')[1]
        print('Executing command: ', cmd)
        if itiba.editor.set_cur_motion(cmd):
            print('Reading states')
            itiba.read_states()
            print('Playing motion')
            itiba.play_cur_motion()


def jtraj_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Joint trajectory with %d points"%(len(data.points)))
    Qs = [p.positions for p in data.points]
    dQs = [p.velocities for p in data.points]
    itiba.read_states()

    # if has_robot:
    #     print('Reading robot joint configuration')
    #     robot.q = panda.get_robot().read_once().q # panda.q
    # else:
    #     robot.q = robot.qr

    print('Planning')
    itiba.plan_joint_trajectory(Qs, dQs, plot=False)

    if has_robot:
        print('Executing')
        itiba.execute()
        print('Execution done')
    else:
        print('Simulating')
        itiba.simulate()
        print('Simulation done')


def main():
    print('TIBA server v0.1')
    print('Waiting for data...')

    rospy.init_node('tiba_server', anonymous=True)
    rospy.Subscriber("/tiba/msg", String, msg_callback)
    rospy.Subscriber("/tiba/jtraj", JointTrajectory, jtraj_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()
