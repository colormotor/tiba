#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState

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
from tiba import cfg

import os, socket, argparse

# Add the --robot argument
parser = argparse.ArgumentParser()
parser.add_argument('--robot', type=str, default='davie', help='Name of the robot (default: tiba)')
# Parse the arguments
args = parser.parse_args()
base_path = './../robots/' + args.robot + '/settings'


def get_ip():
    ''' Gets IP of machine'''
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

if has_robot:
    # hostname = '172.16.0.2'
    # panda = panda_py.Panda(hostname)
    # gripper = libfranka.Gripper(hostname)
    hostname = '172.16.0.2'
    #desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
    desk = panda_py.Desk(hostname, 'admin', 'adminadmin')

    print("Created desk")
    if not desk.has_control():
        print("Forcing control")
        desk.take_control(True)
    desk.activate_fci()
    panda = panda_py.Panda(hostname)
    gripper = libfranka.Gripper(hostname)
else:
    gripper = None
    desk = None
    panda = None



circle_button = FrankaButton('circle')

itiba = TibaInterface(panda, gripper, base_path=base_path)
itiba.safe_mode = False
#robot.q = robot.qr

def on_event(event):
    print(event)
    print('Grasped: ', itiba.grasped)
    if circle_button(event):
        if not itiba.grasped:
            print('Grasp')
            itiba.grasp()
        else:
            print('Ungrasp')
            itiba.ungrasp()

desk.listen(on_event)

state = lambda: None
state.running = True
state.executing = False

qpub = None
respub = None
qdonepub = None

def msg_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Msg: %s", data.data)
    # GOTO joint position is handled explicitly with a joint position (jpos_callback)
    if 'exec:' in data.data:
        cmd = data.data.split(':')[1]
        print('Executing command: ', cmd)
        if cmd == 'grasp':
            itiba.grasp()
            send_joint_state()
            respond_done()
        elif cmd == 'ungrasp':
            itiba.ungrasp()
            send_joint_state()
            respond_done()
        elif cmd == 'programming':
            print('Setting programming mode')
            itiba.set_programming_mode(desk, True)
        elif cmd == 'execution':
            print('Setting execution mode')
            itiba.set_programming_mode(desk, False)

        # if itiba.editor.set_cur_motion(cmd):
        #     print('Reading states')
        #     itiba.read_states()
        #     print('Playing motion')
        #     state.executing = True
        #     itiba.play_cur_motion()
        #     state.executing = False
        #     send_joint_state()

    if 'start:' in data.data:
        itiba.read_states()
        send_joint_state()

    if 'getq' in data.data:
        print("Received request for joint state")
        send_joint_state()

    #respub.publish("Received message")

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
        itiba.execute(force_drawing=False)
        state.executing = False
        print('Execution done')
    else:
        print('Simulating')
        itiba.simulate()
        print('Simulation done')

    send_joint_state()
    respond_done()

def respond_done():
    print("Respondig done")
    itiba.read_states()
    p = JointState()
    p.name = 'Done'
    p.position = itiba.robot.q
    qdonepub.publish(p)
    
    #respub.publish("Done")

def jpos_callback(data):
    q = data.position
    print("Received jpos", q)
    state.executing = True
    itiba.goto_joint_position(q)
    state.executing = False
    send_joint_state()
    respond_done()

def send_joint_state(verbose=False, name='q'):
    itiba.read_states()
    p = JointState()
    p.name = name
    p.position = itiba.robot.q
    #p.name = "Tiba joint state"
    #p = JointTrajectoryPoint()
    #p.positions = itiba.robot.q
    if verbose:
        print('Sending joint state')
    qpub.publish(p)
    #respub.publish("YO this is a very long message why not the joint pub fuck cuckasdas asd asdasd asd asdasdasd asd asd asd asdasd ad")
    if verbose:
        print('Finish sending joint state')

def joint_pos_sender_thread():
    time.sleep(0.5);
    while state.running:
        #respub.publish("YO")
        if not state.executing:
            send_joint_state(False)
        time.sleep(0.1)
    print('Joint thread ended')

def main():
    print('__________________________________________')
    print('o-O-o o-O-o o--o    O')
    print('  |     |   |   |  / \\')
    print('  |     |   O--o  o---o')
    print('  |     |   |   | |   |')
    print('  o   o-O-o o--o  o   o')


    print('\nTIBA server v0.1')
    print('__________________________________________')
    print("Running server on: ", os.environ['ROS_IP'])
    print("Make sure this is reflected in rosmaster.sh")
    print('On client run: source rosmaster.sh')
    print('Waiting for data...')
    print('__________________________________________')

    global qpub, respub, qdonepub

    rospy.init_node('tiba_server', anonymous=True)
    rospy.Subscriber("/tiba/msg", String, msg_callback)
    rospy.Subscriber("/tiba/jtraj", JointTrajectory, jtraj_callback)
    rospy.Subscriber("/tiba/jpos", JointState, jpos_callback)
    #qpub = rospy.Publisher("/tiba/q", JointTrajectoryPoint, queue_size=10) #, queue_size=1)
    qpub = rospy.Publisher("/tiba/q", JointState, queue_size=10) #, queue_size=1)
    qdonepub = rospy.Publisher("/tiba/qdone", JointState, queue_size=10) #, queue_size=1)
    respub = rospy.Publisher("/tiba/response", String, queue_size=10)

    print("Publishing")
    respub.publish("Server started")

    send_joint_state()

    thread = threading.Thread(target=joint_pos_sender_thread)
    thread.start()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    state.running = False
    if desk._listening:
        desk.stop_listen()
    
if __name__ == '__main__':
    main()
