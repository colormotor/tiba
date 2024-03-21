'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

interface - Abstracts the interface with the Franka Emika robot
Uses PandaPy (https://github.com/JeanElsner/panda-py) as a layer on top of libfranka
'''

import importlib

import numpy as np
import os, sys, time
import traceback
import yaml

import roboticstoolbox as rtb
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm

from collections import namedtuple
import swift

from . import ilqr, config, drawing_pad, robot
from . import trajectory_utils as tu
from . import utils, geom, ui
from . import matplotlyb as ply


from importlib import reload
reload(ilqr)
reload(drawing_pad)

cfg = config.cfg

state = lambda: None
state.has_ros = False

rosloader = importlib.find_loader('rospy')
# Load ROS if available, not strictly necessary if not using robot
if rosloader is not None:
    print("Loading ros")
    import rospy
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from std_msgs.msg import Time
    from std_msgs.msg import Header
    from std_msgs.msg import Duration
    state.has_ros = True



class FrankaButton:
    ''' Keeps track of the state of a robot button, given its name'''
    def __init__(self, button='circle'):
        self.button = button
        self.state = False

    def __call__(self, event):
        if not self.button in event:
            return False
        res = False
        if not self.state and event[self.button]:
            res = True
        self.state = event[self.button]
        return res

class MotionEditor:
    def __init__(self, path='motions_desk.json', positions_path='key_positions.json'):
        try:
            self.motions = utils.load_json(path)
        except:
            self.motions = {}

        try:
            self.positions = utils.load_json(positions_path)
        except Exception as e:
            print(e)
            self.positions = {}

        self.path = path
        self.cur_motion = -1
        self.cur_step = -1

        self.positions_path = positions_path
        self.new_motion_name = 'Insert name'
        self.cur_position = -1

        self.new_motion_name = 'Insert name'

    def get_cur_position(self):
        names = list(self.positions.keys())
        if self.cur_position < 0 or self.cur_position >= len(names):
            return {}
        return self.positions[names[self.cur_position]]

    def get_position(self, name):
        if name not in self.positions:
            return None
        return self.positions[name]

    def get_cur_motion(self):
        names = list(self.motions.keys())
        if self.cur_motion < 0 or self.cur_motion >= len(names):
            return {}
        return self.motions[names[self.cur_motion]]

    def save_positions(self):
        if not self.positions:
            return
        print(self.positions)
        utils.save_json(self.positions, self.positions_path)

    def save_motions(self):
        if not self.motions:
            return
        print(self.motions)
        utils.save_json(self.motions, self.path)

    def add_joint_step(self, q):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'q', 'q':list(q)})
        self.save_motions()

    def add_grip_step(self, mm):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'grip', 'mm':mm})
        self.save_motions()

    def add_ungrip_step(self):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion selected')
            return
        motion['steps'].append({'kind':'ungrip'})
        self.save_motions()

    def add_motion(self, name):
        if not name:
            print("No name")
            return

        if name not in self.motions:
            self.motions[name] = {'steps':[]}
            self.save_motions()
        else:
            print("Motion already exists")

    def delete_motion(self, name):
        if name in self.motions:
            self.motions.pop(name)
            self.cur_motion = -1
            self.save_motions()
        else:
            print("Motion does not exist")

    def set_cur_motion(self, name):
        if name not in self.motions.keys():
            print("Motion %s does not exist"%name)
            self.cur_motion = -1
            return False
        self.cur_motion = list(self.motions.keys()).index(name)
        print('Cur motion set to %s'%list(self.motions.keys())[self.cur_motion])
        return True

    def add_position(self, name, q):
        if not name:
            print("No name")
            return

        if name not in self.positions:
            self.positions[name] = list(q)
            self.save_positions()
        else:
            print("Motion already exists")

    def delete_position(self, name):
        if name in self.positions:
            self.positions.pop(name)
            self.cur_position = -1
            self.save_positions()
        else:
            print("Position does not exist")

    def set_cur_position(self, name):
        if name not in self.positions.keys():
            print("Motion %s does not exist"%name)
            self.cur_position = -1
            return False
        self.cur_position = list(self.positions.keys()).index(name)
        print('Cur position set to %s'%list(self.positions.keys())[self.cur_position])
        return True

    def gui(self, imgui, itiba, show_motions=False):
        names = list(self.motions.keys())
        if self.cur_motion >= len(names):
            self.cur_motion = -1

        if show_motions:
            ui.imgui_text_separator("Recorded Motions")
            clicked, self.cur_motion = imgui.listbox("Motion Label", self.cur_motion, names)
            if clicked:
                print(self.cur_motion)

            if imgui.button("Add Motion"):
                imgui.open_popup('Add motion')

            imgui.set_next_window_size(300, 200)
            if imgui.begin_popup_modal('Add motion').opened:
                imgui.text("Select name")
                changed, self.new_motion_name = imgui.input_text('', self.new_motion_name, 512)

                if imgui.button('OK'):
                    self.add_motion(self.new_motion_name)
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button('Cancel'):
                    imgui.close_current_popup()
                imgui.end_popup()

            if self.get_cur_motion():
                imgui.same_line()
                if imgui.button("Delete"):
                    self.delete_motion(names[self.cur_motion])
                if self.get_cur_motion():
                    imgui.text(names[self.cur_motion])
                    self.motion_gui(imgui, itiba, names[self.cur_motion])

        names = list(self.positions.keys())
        if self.cur_position >= len(names):
            self.cur_position = -1

        ui.imgui_text_separator("Key positions")
        clicked, self.cur_position = imgui.listbox("Position Label", self.cur_position, names)
        if clicked:
            print(self.cur_position)

        if imgui.button("Add Position"):
            imgui.open_popup('Add position')

        if self.get_cur_position():
            imgui.same_line()
            if imgui.button("Del"):
                self.delete_position(names[self.cur_position])

        res = ''
        if self.get_cur_position():
            #print("Selected position")
            imgui.same_line()
            if ui.imgui_color_button('Go'):
                q = self.get_cur_position()
                itiba.goto_joint_position(q)
                res = 'goto:' + names[self.cur_position]

        imgui.set_next_window_size(300, 200)
        if imgui.begin_popup_modal('Add position').opened:
            imgui.text("Select name")
            changed, self.new_motion_name = imgui.input_text('', self.new_motion_name, 512)

            if imgui.button('OK'):
                self.add_position(self.new_motion_name, itiba.robot.q)
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

        return res #

    def motion_gui(self, imgui, itiba, key):
        motion = self.motions[key]
        names = []
        for i, step in enumerate(motion['steps']):
            names.append('%d: %s'%(i+1, step['kind']))
        clicked, self.cur_step = imgui.listbox("Steps", self.cur_step, names)
        if clicked:
            print(self.cur_motion)
        if -1 < self.cur_motion < len(names):
            if imgui.button("Delete"):
                motion['steps'].pop(self.cur_step)
                self.save_motions()
        if imgui.button('Play'):
            imgui.open_popup('Play motion')
        if imgui.begin_popup_modal('Play motion').opened:
            imgui.text('Are you sure?')
            if imgui.button('OK'):
                itiba.play_cur_motion()
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

    def get_cur_motion_trajectory(self, q0):
        motion = self.get_cur_motion()
        if not motion:
            print('No motion')
        Qs = []
        dQs = []

        for step in motion['steps']:
            if step['kind'] == 'q':
                Qs_batch, dQs_batch = tu.joint_approach(q0, step['q'])
                q0 = step['q']
                Qs.append(Qs_batch)
                dQs.append(dQs_batch)

        return list(np.vstack(Qs)), list(np.vstack(dQs))

    def draw_steps(self, v3d, robot, pos, size, width):
        v3d = v3d.sketch_view

        motion = self.get_cur_motion()
        if not motion:
            return
        x, y = pos
        c = v3d.canvas()
        v3d.save_viewport()

        for i, step in enumerate(motion['steps']):
            if x + size > width:
                x = pos[0]
                y += size

            rect = geom.make_rect(x, y, size, size)
            rect = geom.scale_rect(rect, 0.95)
            if i == self.cur_step:
                c.stroke(0, 0.5, 1.0)
            else:
                c.stroke(0.5)
            c.fill(0.2, 0.5)
            c.rectangle(rect)
            v3d.set_viewport(rect)
            if step['kind'] == 'q':
                q = step['q']
                P, penP = robot.robot_polylines(q)
                v3d.color(0, 0.5, 1.0)
                v3d.line_width(3)
                v3d.polyline(P)
                v3d.polyline(penP)
            elif step['kind'] == 'grip':
                c.fill(1, 0, 0)
                c.circle(x + size/2, y+size/2, size/4)
            else:
                c.no_fill()
                c.stroke(1, 0, 0)
                c.circle(x + size/2, y+size/2, size/4)

            x += size
        v3d.restore_viewport()

class Motion:
    def __init__(self, Qs, dQs, grasp=None, is_drawing=False):
        self.Qs = list(Qs)
        self.dQs = list(dQs)
        self.grasp = grasp
        self.is_drawing = is_drawing

class TibaInterface:
    def __init__(self, panda, gripper,
                 motion_path='motions_desk.json',
                 pad_path='pad.pkl',
                 scripts_file='./motions/motions.yaml',
                 plot_path='trajectory.png'):

        self.editor = MotionEditor(motion_path)
        self.pad = drawing_pad.DrawingPad(pad_path)
        self.pad.load()
        self.pad.update()

        self.robot = robot.TibaRobot()
        self.collision_robot = robot.TibaRobot()
        self.grasped = False

        self.plot_path = plot_path
        self.panda = panda
        self.gripper = gripper
        self.grasping_state = False

        self.motions = []
        #self.Qs = []
        #self.dQs = []

        self.cup_pos = np.array([0.1176777, -0.40658198, 0.10431843])

        # Assume not in proramming mode
        self.programming = False

        # Load scripted motion path
        self.scripts = {}
        self.scripts_file = scripts_file
        self.reload_scripts()

        print("#################################################")
        print("STARTING UP ROS, IF THIS HANGS CTRL-C TO CONTINUE")
        if state.has_ros:
            try:
                self.msg_pub = rospy.Publisher('/tiba/msg', String, queue_size=10)
                self.jtraj_pub = rospy.Publisher('/tiba/jtraj', JointTrajectory, queue_size=10)
                # rospy.Subscriber('/tiba/response', String, self.response_callback)
                # rospy.Subscriber('/tiba/q', JointState, self.state_callback)
                # rospyrospy.init_node('motion_generator', log_level=rospy.ERROR, anonymous=True) #
                # self.send_msg('start: Tiba sketch connecting')
            except rospy.exceptions.ROSException as e:
                print("Ros issue")
                state.has_ros = False

    def has_ros(self):
        return state.has_ros

    def send_msg(self, data):
        print("Sending msg", data)
        self.msg_pub.publish(data)

    def send_joint_trajectory(self, Qs=None, dQs=None):
        if Qs is None:
            Qs = self.Qs
        if dQs is None:
            dQs = self.dQs
        jtraj = JointTrajectory()
        jtraj.header = Header()
        jtraj.header.stamp = rospy.Time.now()
        jtraj.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6',
                                'panda_joint7']
        for i, (q, dq) in enumerate(zip(Qs, dQs)):
            p = JointTrajectoryPoint()
            p.positions = q
            p.velocities = dq
            #p.time_from_start = Duration(0, int(1e+9*cfg.robot_dt*i))
            jtraj.points.append(p)
        self.jtraj_pub.publish(jtraj)
        rospy.loginfo("Sent joint trajectory")

    def response_callback(self, data):
        print("Received response: ", data.data)

    def state_callback(self, data):
        self.robot.q = np.array(data.position)


    def launch_simulator(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self.robot)

    def clear(self):
        self.motions = []
        #self.Qs = []
        #self.dQs = []

    def reload_scripts(self):
        if os.path.exists(self.scripts_file):
            with open(self.scripts_file, 'r') as f:
                try:
                    self.scripts = yaml.safe_load(f)
                except Exception as e:
                    print(e)
                    #self.scripts = {}
        else:
            self.scripts = {}

    @property
    def Qs(self):
        return sum([m.Qs for m in self.motions], [])

    @property
    def dQs(self):
        return sum([m.dQs for m in self.motions], [])

    def set_tool(self, tool_dict):
        self.clear()
        self.robot = robot.TibaRobot(**tool_dict)
        self.robot.q = self.robot.home
        self.collision_robot = robot.TibaRobot(**tool_dict)
        self.collision_robot.q = self.collision_robot.home


    def set_programming_mode(self, desk, flag):
        ''' Set the robot in programming mode
        Urls adapted from https://github.com/TimSchneider42/franky/blob/321a6b3c5cc8fc49b1e4fae9dde0cf130bb03657/franky/robot_web_session.py
        local Desk must have control with:
        desk = panda_py.Desk(hostname, 'colormotor', 'Rapidograph22!')
        if not desk.has_control():
            desk.take_control(True)
            desk.activate_fci()
        '''
        if flag:
            url = '/desk/api/operating-mode/programming'
        else:
            url = '/desk/api/operating-mode/execution'
        desk._request('post',
                  url,
                  files={'force': True},
                  headers={'X-Control-Token': desk._token.token})

    def simulate(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self.robot)
        t = time.time()
        q_old = self.robot.q
        i = 0
        # while i < len(self.Qs):
        #     self.robot.q = self.Qs[i]

        #     dt = time.time() - t
        #     env.step(cfg.robot_dt)
        #     i += max(1, int(dt/cfg.robot_dt))
        #     t = time.time()
        for motion in self.motions:
            for q in motion.Qs:
                self.robot.q = q
                env.step(cfg.robot_dt)

        # Reset robot after simulation
        self.robot.q = q_old

    def delay(self, delay):
        if delay > 0:
            print('%f second delay'%delay)
            time.sleep(delay)

    def motion_gui(self, imgui):
        return self.editor.gui(imgui, self)

    def read_states(self):
        if self.panda is None:
            # self.robot.q = self.robot.qr
            print("No panda, can't read states")
            return

        self.robot.q = self.panda.get_robot().read_once().q # panda.q
        self.grasped = self.gripper.read_once().is_grasped

    def _exec_joint_motion(self, motion):
        from panda_py import controllers

        Q = motion.Qs
        dQ = motion.dQs

        print('Moving to initial position')
        self.panda.move_to_joint_position(Q[0])
        self.delay(0.15)
        i = 0
        print('Starting control loop')
        ctrl = controllers.JointPosition()
        Kp = np.array([600, 600, 600, 600, 250, 150, 50]) #*0.5
        # Kp = np.array([600, 600, 600, 600, 250, 50, 50]) #*0.5
        #Kp = np.array([600, 600, 600, 600, 250, 20, 50]) #*0.5
        #Kp = np.array([1000.0, 1000.0, 1000.0, 1000.0, 500.0, 300.0, 100.0]) #*0.8
        Kv = 0.5*np.sqrt(2.0*Kp)
        ctrl.set_stiffness(Kp)
        ctrl.set_damping(Kv)
        panda = self.panda
        dq_thresh = 0.001
        dq0 = np.zeros(7)

        panda.start_controller(ctrl)
        with panda.create_context(frequency=1000) as ctx:
            while ctx.ok():
                if i < len(dQ):
                    ctrl.set_control(Q[i], dQ[i])
                    i += 1
                else:
                    ctrl.set_control(Q[-1], dq0)
                    dq = panda.get_state().dq
                    if abs(np.max(dq)) < dq_thresh:
                        break

        print('Stopping controller')
        self.panda.stop_controller()

    def _exec_drawing_motion(self, motion):
        from panda_py import controllers

        Q = motion.Qs
        dQ = motion.dQs

        print('Moving to initial position')
        self.panda.move_to_joint_position(Q[0])

        self.delay(0.5)
        print('Starting cartesian impedance control loop')
        ctrl = controllers.JointPosition()
        Kp = np.diag([800, 800, 50, 10, 10, 10])
        stiffness_orientation = self.pad.get_pen_orientation(self.robot, Q[0]) #self.robot.q)
        ctrl = controllers.CartesianImpedance(impedance=Kp, filter_coeff=1.0,
                                              stiffness_orientation=stiffness_orientation)

        # Get trajectory in 'hand space'
        X = [self.robot.hand_tcp_pos(q) for q in Q]
        n = len(X)
        i = 0
        dq_thresh = 0.001

        panda = self.panda
        panda.start_controller(ctrl)
        with panda.create_context(frequency=1000) as ctx:
            while ctx.ok():
                if i < len(n):
                    ctrl.set_control(X[i][:3], X[i][3:], Q[i])
                    i += 1
                else:
                    ctrl.set_control(X[-1][:3], X[-1][3:], Q[-1])
                    dq = panda.get_state().dq
                    if abs(np.max(dq)) < dq_thresh:
                        break
        print('Stopping cartesian impedance controller')
        self.panda.stop_controller()

    def execute(self, delay=2.0, force_drawing=False):
        ''' Execute the current joint motion'''
        if self.panda is None:
            print("No panda, can't execute")
            return

        self.delay(delay)

        for motion in self.motions:
            if motion.grasp is not None:
                if motion.grasp == False:
                    self.ungrasp()
                elif type(motion.grasp) == bool:
                    self.grasp()
                else:
                    self.grasp(motion.grasp)

            #Q = self.Qs
            #dQ = self.dQs
            if motion.is_drawing or force_drawing:
                self._exec_drawing_motion(motion)
            else:
                self._exec_joint_motion(motion)

            #time.sleep(0.5)
            print('Finished motion')
        return q

    def play_cur_motion(self, delay=2.0):
        ''' Play the currently selected motion'''
        motion = self.editor.get_cur_motion()
        if not motion:
            print('No motion')
            return
        self.play_motion(motion, delay)

    def goto_joint_position(self, q):
        print('goto ', q)
        if self.panda is not None:
            print("Actually moving the robot")
            self.panda.move_to_joint_position(np.array(q))
        else:
            self.robot.q = q

    def play_motion(self, motion, delay=2.0):
        ''' Play a pre-recorded motion'''
        self.delay(delay)

        print('Playing')

        for step in motion['steps']:
            if step['kind'] == 'q':
                #Qs_batch, dQs_batch = tu.joint_approach(self.robot.q, step['q'])
                #Qs_batch = [Qs_batch[0]]*10 + Qs_batch + [Qs_batch[-1]]*10
                #dQs_batch = [dQs_batch[0]]*10 + dQs_batch + [dQs_batch[-1]]*10

                # self.plan_joint_trajectory(Qs_batch, dQs_batch)
                # self.robot.q = self.execute(0.0)
                self.robot.q = step['q']
                if self.panda is not None:
                    self.panda.move_to_joint_position(np.array(step['q']))
            elif step['kind'] == 'grip':
                mm = step['mm']
                if self.panda is not None:
                    self.read_states()
                    if not self.grasped: #not state.graspedt
                        self.grasp(mm)
                    #state.grasped = True

            elif step['kind'] == 'ungrip':
                if self.panda is not None:
                    self.read_states()
                    if self.grasped:
                        self.ungrasp()
                # if state.grasped:
                #     gripper.stop()
                #     gripper.move(100.0/1000, 0.2)
                #     state.grasped = False

        print('Ending')


    def grasp(self, mm=cfg.default_grasp_mm):
        ''' Grasp with a width specified in mm'''
        if self.gripper is None:
            print('No gripper!')
        self.gripper.grasp(mm/1000, 0.2, 10, 0.04, 0.04) # Might want to increase pressure current 10 N

    def ungrasp(self):
        ''' Release the gripper'''
        if self.gripper is None:
            print('No gripper!')
        self.gripper.stop()
        self.gripper.move(100.0/1000, 0.2)

    def plot_trajectory_stats(self, task_trajectories=[]):
        Qs = []
        dQs = []
        for motion in self.motions:
            Qs += motion.Qs
            dQs += motion.dQs

        if Qs:
            ilqr.plot_trajectory_stats(self.robot,
                                            Qs,
                                            dQs,
                                            trajectories=task_trajectories,
                                            filename=self.plot_path)

    def plan_joint_approach(self, q):
        ''' Plan a joint approach to a target position'''
        Qs, dQs = tu.joint_approach(self.robot.q, q)
        self.motions = [Motion(Qs, dQs)]

        
    def plan_joint_trajectory(self, Qs, dQs, plot=False, task_trajectories=[]):
        self.read_states()
        self.motions = [Motion(Qs, dQs)]
        #utils.save_pkl([self.Qs, self.dQs], 'trajectory.pkl')
        # if plot:
            # self.plot_trajectory_stats(task_trajectories)
            # ilqr.plot_trajectory_stats(self.robot,
            #                             self.Qs,
            #                             self.dQs,
            #                             trajectories=task_trajectories,
            #                             filename=self.plot_path)

    def plan_pad_trajectories(self, trajectories, rest=0.0):
        trajectories = self.pad.augment_with_pen_orientation(trajectories, self.robot, self.robot.qr)
        self.plan_task_trajectories(trajectories, rest)

    def compute_task_trajectories(self, trajectories, rest=0.0, q=None, tip_frame='tiba_pen_tcp'):
        ''' Plan trajectories for the robot given in task space as
        a sequence of position and quaternion rotation poses [x, y, z, q_0, q_1, q_2, q_3].
        The trajectory is assumed to be sampled at a frequency of (1.0 / cfg.dt) Hz.
        The result is a joint trajectory sampled at (1.0 / cfg.robot_dt) Hz
        '''
        if q is None:
            q = self.robot.q
        planner = ilqr.iLQRPlannerVelocity(self.robot.urdf_file,
                                           q,
                                           subsample=int(cfg.dt/cfg.robot_dt),
                                           tip_frame=tip_frame
                                           )
        Qs, dQs, linked_trajectories = planner.plan(trajectories, rest=rest, limits=robot.panda_limits, get_trajectories=True)


        # planner = ilqr.iLQRPlannerAcceleration(self.robot.urdf_file,
        #                                        q,
        #                                        subsample=int(cfg.dt/cfg.robot_dt),
        #                                        tip_frame=tip_frame
        #                                        )
        # Qs, dQs, linked_trajectories = planner.plan(trajectories, rest=rest, limits=robot.panda_limits, qd_lim=self.robot.qd_lim, get_trajectories=True)

        plot_cartesian_trajectories(linked_trajectories)
        return Qs, dQs

    def plan_task_trajectories(self, trajectories, rest=0.0, q=None, tip_frame='tiba_pen_tcp'):
        Qs, dQs = self.compute_task_trajectories(trajectories, rest, q, tip_frame)
        self.plan_joint_trajectory(Qs, dQs, trajectories)

    def plan_script(self, path, **kwargs):
        # Local functions
        import numpy as np

        # Local state to incrementally build trajectory
        state = lambda: None
        state.q = self.robot.q
        state.pose = self.robot.pen_tcp_pose(state.q)
        state.hand_pose = self.robot.hand_tcp_pose(state.q)

        # Clear joint trajectory
        self.motions = [Motion([], [])]

        # self.Qs = []
        # self.dQs = []

        print(kwargs)
        # local variable context for script (updated later)
        var_context = {}

        def require_args(*args):
            missing = []
            for arg in args:
                if arg not in kwargs:
                    missing.append(arg)
            if missing:
                raise(ValueError("Missing arguments: " + ", ".join(missing)))

        def update_state(q):
            state.q = q
            state.pose = self.robot.pen_tcp_pose(state.q)
            state.hand_pose = self.robot.hand_tcp_pose(state.q)
            print('Updated pose to', state.pose)

        def joint_move(q):
            if type(q) == str:
                name = q
                q = self.editor.get_position(name)
                if q is None:
                    print(self.editor.positions)
                    raise ValueError("Position does not exist")
            Q, dQ = tu.joint_approach(state.q, q)
            # print(Q)
            # print(dQ)
            self.motions[-1].Qs += list(Q)
            self.motions[-1].dQs += list(dQ)
            update_state(self.motions[-1].Qs[-1])

        def tool_align(y):
            dest_pose = tu.align_pose_zy(state.pose, [0, 0, -1], [-1, 0, 0])
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def tool_align_dip():
            tool_align([-1, 0, 0])

        def hand_align(y):
            dest_pose = self.robot.get_pos(state.q, 'tiba_hand_tcp')
            dest_pose = tu.align_pose_zy(dest_pose, [0, 0, -1], y)
            #dest_pose = self.robot.hand_frame_to_pen_tcp(dest_pose)
            trajectories = [tu.approach_sequence([state.hand_pose, dest_pose])]
            print('Initial position', state.pose)
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q, tip_frame='tiba_hand_tcp')
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def hand_align_pick_tool():
            hand_align([0, -1, 0])

        def move_xyz(v):
            #print(state.pose)
            dest_pose = np.array(state.pose)
            dest_pose[:3, 3] += v
            #print(dest_pose)
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def goto_xyz_hand(pos):
            #print(state.pose)
            dest_pose = np.array(state.hand_pose)
            dest_pose[:3, 3] = pos
            trajectories = [tu.approach_sequence([state.hand_pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q, tip_frame='tiba_hand_tcp')
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def goto_xyz(pos):
            #print(state.pose)
            dest_pose = np.array(state.pose)
            dest_pose[:3, 3] = pos
            #print(dest_pose)
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def xy_circles(radius_mm, rep):
            p0 = state.pose.t
            pts = [p0]
            n = max(5, int(4*np.pi*radius_mm)) # 0.5 mm steps
            radius = radius_mm/1000
            for i in range(rep):
                for a in np.linspace(0, 2*np.pi, n)[:-1]:
                    pts.append(p0 + np.array([np.cos(a), np.sin(a), 0])*radius)
            pts.append(p0)

            paths, _, _ = tu.plan_polylines_time_optimal([np.array(pts)],
                                                                  max_speed=cfg.cartesian_speed,
                                                                  accel_time=cfg.acceleration_time,
                                                                  tol=0.01)
            orn = tu.to_pos_orn(state.pose)[3:]
            trajectories = [[np.concatenate([p, orn]) for p in P] for P in paths]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            self.motions[-1].Qs += list(Qs)
            self.motions[-1].dQs += list(dQs)
            update_state(self.motions[-1].Qs[-1])

        def grasp(mm=cfg.default_grasp_mm):
            self.motions.append(Motion([], [], grasp=mm))

        def ungrasp():
            self.motions.append(Motion([], [], grasp=False))

        print("Compiling script")
        # update varaibles for script
        var_context.update(locals())
        var_context.update(kwargs)
        # Try compiling and executing
        try:
            prog_text = open(path).read()
            prog = compile(prog_text, path, 'exec')
            exec(prog, var_context)
            # Success, view joint trjectory
        except Exception as e:
            print('Error in script: ' + path)
            print(e)
            traceback.print_exc()


def plot_cartesian_trajectories(trajectories):
    return

    ply.figure(title='Cartesian trajectory')

    Tv = 0.0
    Ta = 0.0
    trajectories = [np.array(P) for P in trajectories]

    for i, X in enumerate(trajectories):
        velocity = np.diff(X, axis=0) / cfg.dt
        acceleration = np.diff(velocity, axis=0) / cfg.dt
        #speed = velocities[i] #tu.compute_speed(X, cfg.dt)
        #accel = accelerations[i] #tu.compute_accel(X, cfg.dt)
        tv = np.linspace(0, (len(velocity)-1)*cfg.dt, len(velocity)) + Tv
        ta = np.linspace(0, (len(acceleration)-1)*cfg.dt, len(acceleration)) + Ta
        ply.subplot(1, 1, title='Velocity')
        ply.plot(tv, velocity[:,0], color=ply.categorical_color(i))
        ply.plot(tv, velocity[:,1], color=ply.categorical_color(i))
        ply.plot(tv, velocity[:,2], color='b', alpha=0.5)
        ply.subplot(1, 2, title='Acceleration')
        ply.plot(ta, acceleration[:,0], color=ply.categorical_color(i))
        ply.plot(ta, acceleration[:,1], color=ply.categorical_color(i))
        ply.plot(ta, acceleration[:,2], color='b', alpha=0.5)
        Tv = tv[-1] + cfg.dt
        Ta = ta[-1] + cfg.dt

    ply.subplot(2, 1, title='Trajectory 2d')
    ply.stroke(trajectories, 'k')
    ply.axis_equal(True)

    ply.subplot(2, 2, title='Trajectory 3d')
    for X in trajectories:
        ply.stroke3d(X)
    ply.show()
