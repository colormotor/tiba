#!/usr/bin/env python3
import importlib

import numpy as np
import os, sys, time

from polygonsoup import utils, geom

import roboticstoolbox as rtb
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm

from collections import namedtuple
import swift

from . import planner, config
from . import trajectory_utils as tu

cfg = config.cfg

class FrankaButton:
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
    def __init__(self, path='motions_desk.json'):
        try:
            self.motions = utils.load_json(path)
        except:
            self.motions = {}
        self.path = path
        self.new_motion_name = ''
        self.cur_motion = -1
        self.cur_step = -1

    def get_cur_motion(self):
        names = list(self.motions.keys())
        if self.cur_motion < 0 or self.cur_motion >= len(names):
            return {}
        return self.motions[names[self.cur_motion]]

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

    def gui(self, imgui, itiba):
        names = list(self.motions.keys())
        if self.cur_motion >= len(names):
            self.cur_motion = -1

        clicked, self.cur_motion = imgui.listbox("Motions", self.cur_motion, names)
        if clicked:
            print(self.cur_motion)

        if imgui.button("Add"):
            imgui.open_popup('Add motion')

        if imgui.begin_popup_modal('Add motion').opened:
            imgui.text("Select name")
            changed, self.new_motion_name = imgui.input_text('Name', self.new_motion_name, 512)

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


class TibaInterface:
    def __init__(self, robot, panda, gripper,
                 motion_path='motions_desk.json',
                 pad_path='pad.pkl',
                 plot_path='trajectory.png'):
        self.editor = MotionEditor(motion_path)
        self.pad = planner.DrawingPad(pad_path)
        self.pad.load()
        self.pad.update()

        self.robot = robot
        self.grasped = False

        self.plot_path = plot_path
        self.panda = panda
        self.gripper = gripper
        self.grasping_state = False
        self.Qs = []
        self.dQs = []

        self.cup_pos = np.array([0.1176777, -0.40658198, 0.10431843])

    def launch_simulator(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self.robot)

    def simulate(self):
        env = swift.Swift()
        env.launch(realtime=True)
        env.add(self.robot)

        for q in self.Qs:
            self.robot.q = q
            env.step(cfg.robot_dt)

    def delay(self, delay):
        if delay > 0:
            print('%f second delay'%delay)
            time.sleep(delay)

    def motion_gui(self, imgui):
        self.editor.gui(imgui, self)

    def read_states(self):
        if self.panda is None:
            self.robot.q = self.robot.qr
            print("No panda, can't read states")
            return

        self.robot.q = self.panda.get_robot().read_once().q # panda.q
        self.grasped = self.gripper.read_once().is_grasped

    def execute(self, delay=2.0):
        if self.panda is None:
            print("No panda, can't execute")
            return

        from panda_py import controllers

        self.delay(delay)

        Q = self.Qs
        dQ = self.dQs
        print('Moving to initial position')
        self.panda.move_to_joint_position(Q[0])
        self.delay(0.5)
        i = 0
        print('Starting control loop')
        ctrl = controllers.JointPosition()
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
            q = panda.q
        print('Stopping controller')
        self.panda.stop_controller()
        #time.sleep(0.5)
        print('Finished motion')
        return q

    def play_cur_motion(self, delay=2.0):
        motion = self.editor.get_cur_motion()
        if not motion:
            print('No motion')
            return
        self.play_motion(motion, delay)

    def play_motion(self, motion, delay=2.0):
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


    def grasp(self, mm):
        if self.gripper is None:
            print('No gripper!')
        self.gripper.grasp(mm/1000, 0.2, 10, 0.04, 0.04)

    def ungrasp(self):
        if self.gripper is None:
            print('No gripper!')
        self.gripper.stop()
        self.gripper.move(100.0/1000, 0.2)

    def plan_joint_trajectory(self, Qs, Qd, plot=True, task_trajectories=[]):
        self.read_states()
        Qapproach, dQapproach = tu.joint_approach(self.robot.q, Qs[0])

        self.Qs, self.dQs = Qapproach + Qs, dQapproach + Qd

        utils.save_pkl([self.Qs, self.dQs], 'trajectory.pkl')
        if plot:
            planner.plot_trajectory_stats(self.robot,
                                        self.Qs,
                                        self.dQs,
                                        trajectories=task_trajectories,
                                        filename=self.plot_path)

    def plan_pad_trajectories(self, trajectories, rest=0.05):
        trajectories = self.pad.augment_with_pen_orientation(trajectories, self.robot, self.robot.qr)
        self.plan_task_trajectories(trajectories, rest)

    def plan_task_trajectories(self, trajectories, rest=0.05):
        ilqr = planner.iLQRPlannerVelocity(self.robot.urdf_file,
                                           self.robot.q,
                                           subsample=int(cfg.dt/cfg.robot_dt))
        Qs, dQs = ilqr.plan(trajectories, rest=rest)
        self.plan_joint_trajectory(Qs, dQs, trajectories)
