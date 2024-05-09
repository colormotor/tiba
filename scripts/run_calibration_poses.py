#!/usr/bin/env python3
import importlib
from importlib import reload
from tiba import svg, geom, bezier, spatial, interface, config, utils
import tiba.matplotlyb as ply
import tiba.trajectory_utils as tu
import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import JointState

#os.environ['ROS_IPV6']='off'
#os.environ['ROS_MASTER_URI']='http://158.223.44.103:11311' #Running ROS
#
panda_loader = importlib.find_loader('panda_py')
if panda_loader is not None:
    import panda_py
    from panda_py import libfranka
    from panda_py import controllers
    has_robot = True
else:
    has_robot = False

if has_robot:
    hostname = '172.16.0.2'
    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)
else:
    gripper = None
    panda = None


itiba = interface.TibaInterface(gripper=gripper, panda=panda)
rospy.init_node('robot_joint_state_publisher', anonymous=True)
pub = rospy.Publisher('/tiba/rviz_q', JointState, queue_size=10)
rate = rospy.Rate(10)  # 10 Hz

rec = utils.load_json('./../settings/handeye_poses.json')
poses = rec['poses']

def read_and_send():
    itiba.read_states()
    for i in range(5):
        joint_state = JointState()
        #print('q', itiba.robot.q)
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        joint_state.position = itiba.robot.q
        pub.publish(joint_state)
        rate.sleep()

read_and_send()

input('Tiba calibration poses ready, press enter to go home')
itiba.goto_joint_position('Home')
read_and_send()

for i, q in enumerate(poses):
    input('Pose %d of %d, press enter to go'%(i, len(poses)))
    itiba.goto_joint_position(q)
    read_and_send()

input('Tiba calibration poses over, press enter to go home')
itiba.goto_joint_position('Home')
read_and_send()

print('Goodbye')
