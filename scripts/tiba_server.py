#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import numpy as np
import importlib, time, threading

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

import os, socket

def get_ip():
    # From https://stackoverflow.com/a/28950776
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Turns out we need to tell the node what is the IP to reach this computer
# The trick above seems to do it
os.environ['ROS_IP'] = get_ip()

print("Running server on: ", os.environ['ROS_IP'])
print("Make sure this is reflected in rosmaster.sh")

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

state = lambda: None
state.running = True
state.executing = False

qpub = None
respub = None

def msg_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Msg: %s", data.data)
    if 'exec:' in data.data:
        cmd = data.data.split(':')[1]
        print('Executing command: ', cmd)
        if itiba.editor.set_cur_motion(cmd):
            print('Reading states')
            itiba.read_states()
            print('Playing motion')
            state.executing = True
            itiba.play_cur_motion()
            state.executing = False
            send_joint_state()

    if 'start:' in data.data:
        itiba.read_states()
        send_joint_state()
    respub.publish("Received message")

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
        state.executing = True
        itiba.execute()
        state.executing = False
        print('Execution done')
    else:
        print('Simulating')
        itiba.simulate()
        print('Simulation done')

    send_joint_state()

def send_joint_state(verbose=False):
    itiba.read_states()
    p = JointTrajectoryPoint()
    p.positions = itiba.robot.q
    if verbose:
        print('Sending joint state')
    qpub.publish(p)

def joint_pos_sender_thread():
    time.sleep(0.5);
    while state.running:
        #respub.publish("YO")
        if not state.executing:
            send_joint_state(False)
        time.sleep(0.1)

def main():
    print('TIBA server v0.1')
    print('Waiting for data...')

    global qpub, respub

    rospy.init_node('tiba_server', anonymous=True)
    rospy.Subscriber("/tiba/msg", String, msg_callback)
    rospy.Subscriber("/tiba/jtraj", JointTrajectory, jtraj_callback)
    qpub = rospy.Publisher("/tiba/q", JointTrajectoryPoint, queue_size=10)
    respub = rospy.Publisher("/tiba/response", String, queue_size=10)

    print("Publishing")
    respub.publish("Server started")

    send_joint_state()

    thread = threading.Thread(target=joint_pos_sender_thread)
    thread.start()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    state.running = False

if __name__ == '__main__':
    main()
