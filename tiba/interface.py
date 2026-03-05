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
import runpy
import os, sys, time, copy
import traceback
import yaml
import pdb
import roboticstoolbox as rtb
from spatialmath import SE3, SO3, UnitQuaternion
from spatialmath.base import trnorm
import cv2
from collections import namedtuple
import swift
from PIL import Image
from easydict import EasyDict as edict

from . import ilqr, config, drawing_pad, robot
from . import trajectory_utils as tu
from . import utils, geom, ui
from . import matplotlyb as ply
from .drawing_pad import DrawingPadCalibrator


from importlib import reload
# reload(ilqr)
# reload(drawing_pad)

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

# jrosloader = importlib.find_loader('jupyros')
# if jrosloader is not None:
#     import jupyros as jros

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

class PositionEditor:
    def __init__(self, positions_path):
        try:
            self.positions = utils.load_json(positions_path)
        except Exception as e:
            print(e)
            self.positions = {}

        self.positions_path = positions_path
        self.new_position_name = 'Insert name'
        self.cur_position = -1

        
    def get_cur_position(self):
        names = list(self.positions.keys())
        if self.cur_position < 0 or self.cur_position >= len(names):
            return {}
        return self.positions[names[self.cur_position]]

    def get_position(self, name):
        if name not in self.positions:
            return None
        return self.positions[name]

    def save_positions(self):
        if not self.positions:
            return
        print(self.positions)
        utils.save_json(self.positions, self.positions_path)


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
            changed, self.new_position_name = imgui.input_text('', self.new_position_name, 512)

            if imgui.button('OK'):
                self.add_position(self.new_position_name, itiba.robot.q)
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button('Cancel'):
                imgui.close_current_popup()
            imgui.end_popup()

        return res #

class Motion:
    def __init__(self, Qs, dQs, grasp=None, is_drawing=False):
        self.Qs = list(Qs)
        self.dQs = list(dQs)
        self.grasp = grasp
        self.is_drawing = is_drawing


class WaitForDone:
    def __init__(self, itiba):
        self.itiba = itiba

    def __enter__(self):
        self.itiba.server_busy = True

    def __exit__(self, exception_type, exception_value, exception_traceback):
        print('Waiting for response')
        def print_same_line(s):
            sys.stdout.write("\r" + str(s))
        iter_chars = ['-','/','-','\\']
        i = 0
        while self.itiba.server_busy:
            print_same_line(iter_chars[i%len(iter_chars)])
            time.sleep(0.05)
            i+=1
        resp = self.itiba.response
        if type(resp) != str:
            resp = ''.join(resp)
        if resp.lower() == 'error':
            raise ValueError
        print('Done')

def ros_camera_info_to_cv(cam_info):
    # Camera matrix (3x3)
    camera_matrix = np.array(cam_info.K).reshape(3, 3)

    # Distortion coefficients
    distortion_coefficients = np.array(cam_info.D)

    return camera_matrix, distortion_coefficients


class TibaHandeye:
    def __init__(self, ros_config_path):
        try:
            with open(os.path.expanduser(ros_config_path), 'r') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
                c = self.config['transformation']
                posorn = np.array([c['x'], c['y'], c['z'], c['qw'], c['qx'], c['qy'], c['qz']])
                self.hand_to_camera = tu.to_pose(posorn)
            loaded_config = True
        except FileNotFoundError as e:
            loaded_config = False
            self.hand_to_camera = None
            print(e)
            print('Could not load hand to camera')

        # Init ros subscribers
        self.aruco_frame = None
        self.cam_image = None
        self.cam_info = None
        self.cam_frame = None
        self.aruco_debug = ""
        # Hack for camera misalignment
        self.fudge_offset = [9, 0]
        if state.has_ros and loaded_config:
            print("Loading handeye info")
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import Image as msg_Image
            from visualization_msgs.msg import Marker
            from sensor_msgs.msg import CameraInfo

            def aruco_pose_callback(msg):
                self.aruco_debug = "\nPose callback, no cam frame"

                if self.cam_frame is None:
                    return
                
                self.aruco_debug = "\Pose callback, with cam frame"

                position = msg.pose.position
                orientation = msg.pose.orientation
                posorn = np.array([position.x, position.y, position.z,
                                   orientation.w, orientation.x, orientation.y, orientation.z])
                if self.aruco_frame is None:
                    self.aruco_debug = "\Pose callback, setting frame"
                    self.aruco_frame = self.cam_frame @ np.array(tu.to_pose(posorn))

            if False: #jupyter:
                if jrosloader is None:
                    print("Could not load jupyter ros")
                    rospy_cur = rospy
                else:
                    rospy_cur = jros
            else:
                rospy_cur = rospy

            rospy_cur.Subscriber("/aruco_single/pose", PoseStamped, aruco_pose_callback)
            print("Subscribed to aruco marker")

            def got_image(data):
                im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
                self.cam_image = im
            #print(data.shape)
            rospy_cur.Subscriber('/camera/color/image_raw', msg_Image, got_image)
            print("Subscribed to realsense")

            def camera_info_callback(msg):
                self.cam_info = msg

            rospy_cur.Subscriber('/camera/color/camera_info', CameraInfo, camera_info_callback)
            print("Subscribed to camera info")

    def is_valid(self):
        return (self.cam_info is not None and
                self.hand_to_camera is not None and
                self.cam_image is not None  and
                self.aruco_frame is not None)
        
    def update(self, interface):
        if self.hand_to_camera is None:
            print('No hand to camera')
            return
        q = interface.robot.q
        hand_frame = np.array(interface.robot.fkine(q, start='tiba_link0', end='tiba_hand'))
        self.cam_frame =  hand_frame @ np.array(self.hand_to_camera)
        
    def draw_gl(self, interface, v3d):
        if not self.is_valid():
            return
        
        v3d.frame(self.cam_frame, alpha=0.5, scale=0.1)

        if self.aruco_frame is not None:
            h = 50/1000
            v3d.frame(self.aruco_frame, alpha=1, scale=h)
            
            ax = self.aruco_frame[:3,0]
            ay = self.aruco_frame[:3,1]
            p = self.aruco_frame[:3,3]
            v3d.color(1,0,1)
            pts = [p - ax*h - ay*h,
                   p + ax*h - ay*h,
                   p + ax*h + ay*h,
                   p - ax*h + ay*h]
            v3d.polyline(pts, closed=True)

            vo = 105/1000
            hw = 290/2000
            ph = 410/1000
            pts = [p - ay*vo - ax*hw,
                   p - ay*vo + ax*hw,
                   p - ay*vo + ax*hw - ay*ph,
                   p - ay*vo - ax*hw - ay*ph]
            v3d.polyline(pts, closed=True)

    def project_points(self, pts, space='world', fudge=True):
        pts = np.array(pts)
        inv_cam = np.linalg.inv(self.cam_frame)
        aruco_frame_eye = inv_cam @ self.aruco_frame
        aruco_frame_eye_inv = np.linalg.inv(aruco_frame_eye)
        rot = aruco_frame_eye[:3,:3]
        pos = aruco_frame_eye[:3,3]
        cam_matrix, distortion = ros_camera_info_to_cv(self.cam_info)
        if space=='eye':
            points_eye = np.array(pts)
        elif space=='aruco':
            #print(pts)
            points_eye = np.array([geom.tsm(aruco_frame_eye, p) for p in pts])
        else: # world
            points_eye = np.array([geom.tsm(inv_cam, p) for p in pts])
        points_aruco = geom.tsm(aruco_frame_eye_inv, points_eye)
        projected_points, _ = cv2.projectPoints(points_aruco, rot, pos, cam_matrix, distortion)
        projected_points = [p[0] for p in projected_points]
        if fudge:
            projected_points = [P + self.fudge_offset for P in projected_points]
        return projected_points
    
class TibaInterface:
    def __init__(self, panda=None, gripper=None, init_node='',
                 pad_path='drawing_pad.json',
                 scripts_file='motions/motions.yaml',
                 positions_path='key_positions.json',
                 tools_path='tools.json',
                 plot_path='trajectory.png',
                 handeye_config_path='panda_eye_hand_calib_eye_on_hand.yaml',
                 base_path='./../robots/tiba/settings',
                 drawing_orientation_stiffness=0.7,
                 ):

        if not os.path.isdir(base_path):
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_path)
            print('base path:', base_path)
        self.base_path = base_path
        pad_path = os.path.join(base_path, pad_path)
        scripts_file = os.path.join(base_path, scripts_file)
        handeye_config_path = os.path.join(base_path, handeye_config_path)
        positions_path = os.path.join(base_path, positions_path)
        tools_path = os.path.join(base_path, tools_path)

        self.drawing_orientation_stiffness = drawing_orientation_stiffness
        self.editor = PositionEditor(positions_path)
        self.pad = drawing_pad.DrawingPad(pad_path)
        self.pad.load()
        self.pad.update()

        self.safe_mode = False
        self.trajectory_status = True
        self.selected_tool = None

        self.robot = robot.TibaRobot()
        self.collision_robot = robot.TibaRobot()
        self.pad_calib = DrawingPadCalibrator(self.base_path, self.robot)
        self.robot.q = self.robot.home
        self.debug_q = None
        self.grasped = False
        self.tools_path = tools_path
        self.tool_rack_path = os.path.join(base_path, "tool_rack.json")
        self.tool_rack = {
            "p1": None,        # pose of tool 1 (4x4, list of lists)
            "p2": None,        # pose of tool 2 (4x4, list of lists)
            "spacing": None,   # distance between neighboring tools [m]
            "direction": None, # unit vector from tool 1 to tool 2
            "count": 3,        # number of tools on the rack
        }
        self.color_palette_path = os.path.join(base_path, "color_palette.json")
        self.color_palette = {
            "p00": None,                # pose of tool 00 (4x4, list of lists)
            "p01": None,                # pose of tool 01 (4x4, list of lists)
            "p10": None,                # pose of tool 10 (4x4, list of lists)
            "spacing_in_rows": None,    # distance between colors in a row [m]
            "spacing_in_cols": None,    # distance between colors in a column [m]
            "direction_rows": None,     # unit vector from tool 00 to tool 01
            "direction_cols": None,     # unit vector from tool 00 to tool 10
            "count_rows": 3,            # number of colors in a row
            "count_cols": 2,            # number of colors in a column
        }
        self.tool_rack_selected = 0

        # Try to load existing calibration
        try:
            data = utils.load_json(self.tool_rack_path)
            self.tool_rack.update(data)
            print("[ToolRack] Loaded calibration from", self.tool_rack_path)
        except FileNotFoundError:
            print("[ToolRack] No saved tool rack calibration found")

        try:
            data = utils.load_json(self.color_palette_path)
            self.color_palette.update(data)
            print("[ColorPalette] Loaded calibration from", self.color_palette_path)
        except FileNotFoundError:
            print("[ColorPalette] No saved color palette calibration found")

        self.tool_rack_status = ""
        self.color_palette_step = 0  # 0=idle, 1=00, 2=01, 3=rows, 4=10, 5=cols
        self.color_palette_status = ""
        self.color_row = 0
        self.color_col = 0
        self.color_script = "motions/simple_dip.py"
        self.color_status = ""

        self.plot_path = plot_path
        self.panda = panda
        self.gripper = gripper
        self.grasping_state = False
        self.log = []
        self.motions = []
        #self.Qs = []
        #self.dQs = []
        self.reload_tools()

        self.cup_pos = np.array([0.1176777, -0.40658198, 0.10431843])

        # Assume not in proramming mode
        self.programming = False

        # Load scripted motion path
        self.scripts = {}
        self.scripts_file = scripts_file
        self.reload_scripts()

        self.server_busy = False
        self.response = 'Done'

        self.recorded_Qs = []
        self.recorded_dQs = []
        
        if state.has_ros:
            print("#################################################")
            print("STARTING UP ROS, IF THIS HANGS CTRL-C TO CONTINUE")
            if False: #jupyter:
                if jrosloader is None:
                    print("Could not load jupyter ros")
                    rospy_cur = rospy
                else:
                    rospy_cur = jros
            else:
                rospy_cur = rospy
                print("Started ROS")

            try:
                if init_node and not rospy.core.is_initialized():
                    print("Initializing node", init_node)
                    rospy.init_node(init_node, anonymous=True, log_level=rospy.ERROR)
                    print("Node initialized, rospy name:", rospy.get_name())
                else:
                    print("ROS already initialized, skipping init_node")

                rospy_cur.Subscriber('/tiba/q', JointState, self.state_callback)
                rospy_cur.Subscriber('/tiba/qdone', JointState, self.qdone_callback)
                rospy_cur.Subscriber('/tiba/executed_traj', JointTrajectory, self.executed_traj_callback)
                rospy_cur.Subscriber('/tiba/response', String, self.response_callback)

                print("Creating subscribers and publisher")
                self.msg_pub  = rospy_cur.Publisher('/tiba/msg', String, queue_size=10)
                self.jtraj_pub = rospy_cur.Publisher('/tiba/jtraj', JointTrajectory, queue_size=10)
                self.jpos_pub  = rospy_cur.Publisher('/tiba/jpos', JointState, queue_size=10)

                rospy.sleep(1)
                self.send_msg('start: Tiba sketch connecting')
                # rospy.Subscriber('/tiba/response', String, self.response_callback)
                # rospy.Subscriber('/tiba/q', JointState, self.state_callback)
                # rospyrospy.init_node('motion_generator', log_level=rospy.ERROR, anonymous=True) #
                # self.send_msg('start: Tiba sketch connecting')
                print("Init handeye coordination")
                self.handeye = TibaHandeye(handeye_config_path) #jupyter=jupyter)
            except rospy.exceptions.ROSException as e:
                print("Ros issue, disabling")
                print("  ROSException:", e)
                state.has_ros = False
                self.handeye = None
        else:
            print("No ros")
            self.handeye = None
        print('end interface init')
        #self.ros_thread_running = False
        #self.ros_thread = None

    def __del__(self):
        pass
        # if self.ros_thread is not None:
        #     self.ros_thread_running = False
        #     self.ros_thread.join()

    # def _ros_thread(self):
    #     #rospy.spin()
    #     while not rospy.core.is_shutdown() and self.ros_thread_running:
    #         rospy.rostime.wallsleep(0.1)

    # def subscribe_to_state(self):
    #     rospy.Subscriber('/tiba/q', JointState, self.state_callback)

    #     # from threading import Thread
    #     # if not self.has_ros():
    #     #     return
    #     # print("Starting ros thread")
    #     # self.ros_thread = Thread(target=self._ros_thread)
    #     # self.ros_thread_running = True
    #     # self.ros_thread.start()

    def reload_tools(self):
        self.tools = utils.load_json(self.tools_path)
        print("Reloaded tools")
        print(self.tools)

    def print_available_tools(self):
        self.reload_tools()
        print("Available tools:")
        print(self.tools.keys())

    def print_available_positions(self):
        print("Available key positions:")
        print(self.editor.positions.keys())

    def print_available_scripts(self):
        print("Available scripts:")
        print(self.scripts.keys())

    def handeye_valid(self):
        return self.handeye is not None and self.handeye.is_valid()

    def shutdown(self):
        pass
        #if self.has_ros():
        #    rospy.signal_shutdown('Closing Tiba interface')
            
    def has_ros(self):
        return state.has_ros

    def send_msg(self, data):
        print("send_msg called, has_ros =", self.has_ros())
        print("self.msg_pub:", getattr(self, "msg_pub", None))
        try:
            if not self.has_ros():
                print("send_msg: has_ros False, abort")
                return
            print("Sending msg", data)
            self.msg_pub.publish(data)
            rospy.sleep(0.05)
        except Exception as e:
            print("send_msg ERROR:", e)

    def clear_aruco(self):
        self.update_handeye()
        if self.handeye is not None:
            self.handeye.aruco_frame = None
            
    def update_handeye(self):
        if self.handeye is not None:
            self.handeye.update(self)

    def send_joint_trajectory(self, Qs=None, dQs=None):
        print("Sending trajectory")
        if Qs is None:
            Qs = self.Qs
        if dQs is None:
            dQs = self.dQs

        self.trajectory_status = self.robot.check_joint_trajectory(Qs, dQs)
        if not self.trajectory_status:
            print("Joint trajectory exceeds safety limits, plotting")
            ilqr.plot_trajectory_stats(self.robot,
                                            Qs,
                                            dQs
                                            )

            return

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
        print("Trajectory sent")
        rospy.loginfo("Sent joint trajectory")

    def wait(self):
        return WaitForDone(self)

    def send_joint_pos(self, q):
        p = JointState()
        p.position = q
        self.jpos_pub.publish(p)
        rospy.loginfo("Sent joint position")

    def executed_traj_callback(data):
        rospy.loginfo(rospy.get_caller_id() + "Robot executed joint trajectory with %d points"%(len(data.points)))
        Qs = [p.positions for p in data.points]
        dQs = [p.velocities for p in data.points]
        self.recorded_Qs = Qs
        self.recorded_dQs = dQs
        
    def response_callback(self, data):
        print("Received response: ", data.data)
        # if data.data == 'Done':
        #     self.server_busy = False

    def qdone_callback(self, data):
        self.robot.q = np.array(data.position)
        self.debug_q = np.array(data.position)
        self.server_busy = False
        self.response = data.name

    def state_callback(self, data):
        self.robot.q = np.array(data.position)
        self.debug_q = np.array(data.position)
        #self.debug_state = data.name
        # if 'Done' in data.name:
        #     #print("Received done joint state")
        #     self.server_busy = False

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
        self.reload_tools()
        # Accept key as input
        if type(tool_dict) == str:
            tool_dict = self.tools[tool_dict]

        self.clear()
        # Setting the tool recreates the robot (and URDF)
        self.robot = robot.TibaRobot(**tool_dict)
        self.robot.q = self.robot.home
        self.collision_robot = robot.TibaRobot(**tool_dict)
        self.collision_robot.q = self.collision_robot.home
        # Propagate tool paramaters with defaults
        self.tool_info = {'offset_mm': 0.0,
                          'range_mm': 0.0,
                          'min_radius_mm': 1.0,
                          'max_radius_mm': 1.0,
                          }
        for key, v in tool_dict.items():
            if key in self.tool_info:
                self.tool_info[key] = v
        self.tool_info = edict(self.tool_info)
        # Deprecate this
        # if 'pen_offset_base_mm' in tool_dict:
        #     self.pen_offset_base_mm = tool_dict['pen_offset_base_mm']
        # else:
        #     self.pen_offset_base_mm = 0.0
        # if 'min_radius_mm' in tool_dict:
        #     self.min_radius_mm = tool_dict['min_radius_mm']
        # else:
        #     self.min_radius_mm = 0.0
        # if 'max_radius_mm' in tool_dict:
        #     self.max_radius_mm = tool_dict['max_radius_mm']
        # else:
        #     self.max_radius_mm = 0.0

    def programming_mode(self, flag=True):
        if flag:
            self.send_msg('exec:programming')
        else:
            self.send_msg('exec:execution')
            
    def set_programming_mode(self, desk, flag):
        ''' Set the robot in programming mode (internal)
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
        try:
            desk._request('post',
                  url,
                  files={'force': True},
                  headers={'X-Control-Token': desk._token.token})
        except ConnectionError as e:
            print(e)
            

    def simulate(self):
        ''' Simulate current trajectories using Swift'''
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

    def pad_calibration_gui(self, imgui):
        """
        Guided drawing pad calibration.
        The user should move the TCP to each corner in this order:
        1) bottom left
        2) bottom right
        3) top right
        The 4th (top left) corner is computed automatically in DrawingPadCalibrator.save().
        """
        if self.pad_calib is None:
            imgui.text("Pad calibration not available.")
            return

        corner_names = ["bottom left", "bottom right", "top right"]
        num_points = len(self.pad_calib.points)

        imgui.separator()
        imgui.text("Drawing Pad Calibration")
        imgui.separator()

        # Instruction which corner to set next
        if num_points < len(corner_names):
            next_name = corner_names[num_points]
            imgui.text_wrapped(
                (f"Step {num_points+1}/3: Go to {next_name} corner, press 'Add corner'.")
            )
        else:
            imgui.text_wrapped(
                "All 3 corners set. The 4th (top left) will be computed automatically on save."
            )

        imgui.text(f"Recorded corners: {num_points} / 3")

        if imgui.button("Add current corner"):
            try:
                idx = self.pad_calib.add_current_corner()
                imgui.text(f"Added corner #{idx}")
            except Exception as e:
                imgui.text(f"Error adding corner: {e}")

        imgui.same_line()
        if imgui.button("Reset calibration"):
            self.pad_calib.reset()

        if imgui.button("Save pad calibration"):
            try:
                path = self.pad_calib.save()
                imgui.text(f"Saved pad to: {path}")
            except Exception as e:
                imgui.text(f"Error saving pad: {e}")


    def start_pad_calibration(self):
        """
        Start calibration of the drawing pad.

        After calling this, the user should move the robot to the three
        pad corners in the following order and trigger 'save_pad_corner'
        for each:

            1) bottom left
            2) bottom right
            3) top right
        """
        self.pad_calib.reset()
        print("[TibaInterface] Pad calibration started")
        print(" -> Move the pen to the BOTTOM LEFT corner and send 'exec:calib_pad_corner'")

    def save_pad_corner(self):
        """
        Read the current pen TCP position and append it as a new corner.

        Returns:
            count (int): number of recorded corners after this call.
        Also prints guidance for the next step.
        """
        count = self.pad_calib.add_current_corner()

        if count == 1:
            print("[TibaInterface] Saved corner 1 (bottom left)")
            print(" -> Move to the BOTTOM RIGHT corner and send 'exec:calib_pad_corner'")
        elif count == 2:
            print("[TibaInterface] Saved corner 2 (bottom right)")
            print(" -> Move to the TOP RIGHT corner and send 'exec:calib_pad_corner'")
        elif count == 3:
            print("[TibaInterface] Saved corner 3 (top right)")
            print(" -> Send 'exec:calib_pad_finish' to compute the 4th corner and save the pad")
        else:
            print(f"[TibaInterface] Saved additional corner #{count}")

        return count

    def finish_pad_calibration(self):
        """
        Finalize drawing pad calibration.

        Expects at least three recorded corners, computes the fourth,
        and writes robots/<robot>/settings/drawing_pad.json.
        """
        if not self.pad_calib.is_complete():
            raise RuntimeError(
                "Not enough corners recorded for pad calibration (need 3)"
            )

        path = self.pad_calib.save()
        print(f"[TibaInterface] Pad calibration finished, file: {path}")
        return path


    def _current_tcp_pose(self):
        """Return current pen TCP pose as 4x4 matrix in base frame."""
        pose = self.robot.pen_tcp_pose(self.robot.q)
        return np.array(pose)

    def _save_tool_rack(self):
        """Save current tool rack calibration to JSON."""
        print("[DEBUG] Calling _save_tool_rack, tool_rack keys:", self.tool_rack.keys())
        utils.save_json(self.tool_rack, self.tool_rack_path)
        print("[ToolRack] Saved calibration to", self.tool_rack_path)

    def tool_rack_set_tool1(self):
        """Teach pose of tool 1 from current TCP."""
        self.tool_rack["p1"] = self._current_tcp_pose().tolist()
        self._save_tool_rack()
        print("[ToolRack] Set tool 1 pose")

    def tool_rack_set_tool2(self):
        """Teach pose of tool 2 from current TCP and derive spacing + direction."""
        if self.tool_rack["p1"] is None:
            raise RuntimeError("Tool 1 pose must be set before tool 2")

        p1 = np.array(self.tool_rack["p1"])
        p2 = self._current_tcp_pose()
        print(f"[ToolRack] p2={p2}")
        self.tool_rack["p2"] = p2.tolist()

        x1 = p1[:3, 3]
        x2 = p2[:3, 3]
        diff = x2 - x1
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-6:
            raise RuntimeError("Tool 1 and 2 are at the same position")

        direction = [0, -1, 0]
        print(direction)

        self.tool_rack["spacing"] = dist
        self.tool_rack["direction"] = direction
        self._save_tool_rack()

        print(f"[ToolRack] spacing={dist:.4f} m, dir={direction}")

    def tool_rack_is_calibrated(self):
        return (
                self.tool_rack["p1"] is not None
                and self.tool_rack["spacing"] is not None
                and self.tool_rack["direction"] is not None
        )


    def take_tool(self, index):
        if self.selected_tool is not None:
            print('A tool is already selected')
            print("Returning tool", self.selected_tool)
            self.store_tool(self.selected_tool)
            time.sleep(10)

        print("robot.q =", self.robot.q)
        # 1) Retrieve the approach and reach pose from the rack
        T_approach, T_tool = self._tool_rack_get_approach_pose(index)

        # 2) IK on Pen-TCP
        q_approach = self.robot.ikine_pen_tcp(T_approach)
        q_tool     = self.robot.ikine_pen_tcp(T_tool)

        if q_approach is None or q_tool is None:
            raise RuntimeError("IK failed")

        print("T_approach Z=%.3f, T_tool Z=%.3f" % (T_approach.t[2], T_tool.t[2]))

        # 3) Trajectory: current → approach → tool → approach
        Qs, dQs = [], []

        # Current → Approach (fast)
        Q1, dQ1 = tu.joint_approach(self.robot.q, q_approach)
        Qs += list(Q1); dQs += list(dQ1)

        # Approach → Tool (VERY SLOW for Grip)
        Q2, dQ2 = tu.joint_approach(q_approach, q_tool, geom.radians(5.0))
        Qs += list(Q2)
        dQs += list(dQ2)

        # Publish Trajectory (until Tool)
        self.send_joint_trajectory(Qs, dQs)
        print("Trajectory: current→approach→tool (slow grip phase)")
        time.sleep(10.0)

        # 4) GRASP at the Tool Point (after reaching!)
        print("Executing grasp at tool position...")
        self.send_msg("exec:grasp")
        time.sleep(2.0)

        # 5) Retreat: Tool → Approach (medium fast)
        Q3, dQ3 = tu.joint_approach(q_tool, q_approach)
        self.send_joint_trajectory(list(Q3), list(dQ3))
        print("Retract: tool→approach")
        time.sleep(3.0)

        print("Moving to neutral position (gripper closed)...")
        self.send_msg("exec:home")

        self.selected_tool = index
        self.send_msg(f"took_tool_{index}")


    def store_tool(self, index):
        if self.selected_tool is None:
            print('No tool selected')
            return

        print("robot.q =", self.robot.q)
        # 1) Retrieve the approach and reach pose from the rack
        T_approach, T_tool = self._tool_rack_get_approach_pose(index)
        offset = np.array([0, 0, 0.02])
        T_drop = SE3(offset) * T_tool

        # 2) IK on Pen-TCP
        q_approach = self.robot.ikine_pen_tcp(T_approach)
        q_drop     = self.robot.ikine_pen_tcp(T_drop)


        if q_approach is None or q_drop is None:
            raise RuntimeError("IK failed")

        print("T_approach Z=%.3f, T_tool Z=%.3f" % (T_approach.t[2], T_tool.t[2]))

        # 3) Trajectory: current → approach → tool → approach
        Qs, dQs = [], []

        # Current → Approach (fast)
        Q1, dQ1 = tu.joint_approach(self.robot.q, q_approach)
        Qs += list(Q1); dQs += list(dQ1)

        # Approach → Tool (VERY SLOW for Grip)
        Q2, dQ2 = tu.joint_approach(q_approach, q_drop, geom.radians(3.0))
        Qs += list(Q2)
        dQs += list(dQ2)

        # Publish Trajectory (until Tool)
        self.send_joint_trajectory(Qs, dQs)
        print("Trajectory: current→approach→tool (slow grip phase)")
        time.sleep(12.0)

        # 4) UNGRASP at the Tool Point (after reaching!)
        print("Executing grasp at tool position...")
        self.send_msg("exec:ungrasp")
        time.sleep(2.0)

        # 5) Retreat: Tool → Approach (medium fast)
        Q3, dQ3 = tu.joint_approach(q_drop, q_approach)
        self.send_joint_trajectory(list(Q3), list(dQ3))
        print("Retract: tool→approach")
        time.sleep(3.0)

        print("Moving to neutral position (gripper closed)...")
        self.send_msg("exec:home")
        self.selected_tool = None

        self.send_msg(f"stored_tool_{index}")


    def _tool_rack_get_pose(self, index):
        """SE3-Pose of the Pen-TCP in slot `index`."""
        if self.tool_rack["p1"] is None or self.tool_rack["direction"] is None:
            raise RuntimeError("Tool rack not calibrated")

        T0 = SE3(np.array(self.tool_rack["p1"]))
        direction = np.array(self.tool_rack["direction"])
        direction = direction / np.linalg.norm(direction)

        print("T0.t =", T0.t)
        print("direction =", direction, "spacing =", self.tool_rack["spacing"])
        offset = index * self.tool_rack["spacing"] * direction
        print(f"index={index}, offset={offset}")
        T_tool = T0 * SE3(offset[0], offset[1], offset[2])
        print("T_tool.t =", T_tool.t)
        return T_tool

    def _tool_rack_get_approach_pose(self, index, dist=0.2):
        T_tool = self._tool_rack_get_pose(index)
        offset = np.array([0, 0, dist])
        T_approach = SE3(offset) * T_tool
        return T_approach, T_tool


    def tool_rack_gui(self, imgui):
        imgui.separator()
        imgui.text("Tool rack calibration")
        imgui.separator()

        imgui.text_wrapped(
            "Step 1: Move the pen TCP to the center of tool 1 in the rack and press 'Set tool 1'."
        )
        if imgui.button("Set tool 1"):
            try:
                self.tool_rack_set_tool1()
            except Exception as e:
                imgui.text(f"Error: {e}")

        imgui.text_wrapped(
            "Step 2: Move the pen TCP to the center of tool 2 and press 'Set tool 2'. "
            "Spacing and direction will be computed automatically."
        )
        if imgui.button("Set tool 2"):
            try:
                self.tool_rack_set_tool2()
            except Exception as e:
                imgui.text(f"Error: {e}")

        if self.tool_rack_is_calibrated():
            imgui.text(
                f"Calibrated: spacing={self.tool_rack['spacing']:.3f} m, "
                f"dir={self.tool_rack['direction']}"
            )
        else:
            imgui.text("Calibrated: no")

        # Number of tools in the rack
        changed, count = imgui.input_int("Tool count", int(self.tool_rack["count"]))
        if changed and count > 0:
            self.tool_rack["count"] = int(count)
            self._save_tool_rack()

        # Tool index selection (same style as Tool count)
        changed, idx = imgui.input_int("Tool index", int(self.tool_rack_selected))
        if changed:
            # clamp to valid range
            max_idx = int(self.tool_rack["count"]) - 1
            self.tool_rack_selected = max(0, min(idx, max_idx))

        imgui.same_line()
        if imgui.button("-"):
            if self.tool_rack_selected > 0:
                self.tool_rack_selected -= 1

        imgui.same_line()
        if imgui.button("+"):
            max_idx = int(self.tool_rack["count"]) - 1
            if self.tool_rack_selected < max_idx:
                self.tool_rack_selected += 1


        if imgui.button("Take selected tool"):
            try:
                print("Taking tool", self.tool_rack_selected)
                self.take_tool(self.tool_rack_selected)
                self.tool_rack_status = f"Took tool {self.tool_rack_selected}."
            except Exception as e:
                self.tool_rack_status = f"Error taking tool {self.tool_rack_selected}: {e}"


        imgui.separator()

        # Store tool in rack
        imgui.text("Store tool in rack")
        if imgui.button("Store tool##store"):
            self.selected_tool = self.tool_rack_selected
            if self.selected_tool is None:
                self.tool_rack_status = "No tool selected."
            else:
                try:
                    print("Storing tool", self.selected_tool)
                    self.store_tool(self.selected_tool)
                    self.tool_rack_status = f"Stored tool."
                except Exception as e:
                    self.tool_rack_status = f"Error storing tool {self.selected_tool}: {e}"

        if imgui.button("Clear tool##rack"):
            self.selected_tool = None
            self.tool_rack_status = "Cleared tool."


        if self.tool_rack_status:
            imgui.text_wrapped(self.tool_rack_status)

    def _save_color_palette(self):
        """Save current color palette calibration to JSON."""
        utils.save_json(self.color_palette, self.color_palette_path)
        print("[ToolRack] Saved calibration to", self.color_palette_path)

    def color_palette_set_color_00(self):
        """Teach pose of color 00 from current TCP."""
        self.color_palette["p00"] = self._current_tcp_pose().tolist()
        self._save_color_palette()
        print("[ColorPalette] Set color 00 pose")

    def color_palette_set_color_01(self):
        """Teach pose of color 01 from current TCP and derive spacing + direction."""
        if self.color_palette["p00"] is None:
            raise RuntimeError("Color 00 pose must be set before color 01")

        p00 = np.array(self.color_palette["p00"])
        p01 = self._current_tcp_pose()
        self.color_palette["p01"] = p01.tolist()

        diff = p01[:3, 3] - p00[:3, 3]
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-6:
            raise RuntimeError("Color 00 and 01 are at the same position")

        direction = [-1, 0, 0]

        self.color_palette["spacing_in_rows"] = dist
        self.color_palette["direction_rows"] = direction
        self._save_color_palette()

        print(f"[ColorPalette] spacing={dist:.4f} m, dir={direction}")

    def color_palette_set_color_10(self):
        """Teach pose of color 10 from current TCP and derive spacing + direction."""
        if self.color_palette["p00"] is None:
            raise RuntimeError("Color 00 pose must be set before color 10")

        p00 = np.array(self.color_palette["p00"])
        p10 = self._current_tcp_pose()
        self.color_palette["p10"] = p10.tolist()

        diff = p10[:3, 3] - p00[:3, 3]
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-6:
            raise RuntimeError("Color 00 and 01 are at the same position")

        direction = [0, -1, 0]

        self.color_palette["spacing_in_cols"] = dist
        self.color_palette["direction_cols"] = direction
        self._save_color_palette()

        print(f"[ColorPalette] spacing={dist:.4f} m, dir={direction}")

    def color_palette_is_calibrated(self):
        return (
            self.color_palette["p00"] is not None
            and self.color_palette["spacing_in_rows"] is not None
            and self.color_palette["direction_rows"] is not None
            and self.color_palette["spacing_in_cols"] is not None
            and self.color_palette["direction_cols"] is not None
        )

    def get_color(self, row, col, script_path, **script_args):

        print("robot.q =", self.robot.q)
        # 1) Retrieve the approach and reach pose from the rack
        T_approach, T_tool = self._color_palette_get_approach_pose(row, col)

        # 2) IK on Pen-TCP
        q_approach = self.robot.ikine_pen_tcp(T_approach)
        q_tool     = self.robot.ikine_pen_tcp(T_tool)

        if q_approach is None or q_tool is None:
            raise RuntimeError("IK failed")

        print("T_approach Z=%.3f, T_tool Z=%.3f" % (T_approach.t[2], T_tool.t[2]))

        # 3) Trajectory: current → approach → tool → approach
        Qs, dQs = [], []

        # Current → Approach (fast)
        Q1, dQ1 = tu.joint_approach(self.robot.q, q_approach)
        Qs += list(Q1); dQs += list(dQ1)
        self.send_joint_trajectory(Qs, dQs)
        print("Trajectory: current→approach")
        time.sleep(5)
        script_args["T_tool"] = T_tool
        script_args["T_approach"] = T_approach

        self.plan_and_execute_script(script_path, **script_args)
        time.sleep(5)

    def _color_palette_get_pose(self, row, col):
        """SE3-Pose of the Pen-TCP at color cell (row, col)."""
        if not self.color_palette_is_calibrated():
            raise RuntimeError("Color palette not calibrated")

        T00 = SE3(np.array(self.color_palette["p00"]))

        dir_rows = np.array(self.color_palette["direction_rows"])
        dir_rows = dir_rows / np.linalg.norm(dir_rows)

        dir_cols = np.array(self.color_palette["direction_cols"])
        dir_cols = dir_cols / np.linalg.norm(dir_cols)

        spacing_rows = float(self.color_palette["spacing_in_rows"])
        spacing_cols = float(self.color_palette["spacing_in_cols"])

        # offset = col * row-spacing + row * col-spacing
        offset = row * spacing_rows * dir_rows + col * spacing_cols * dir_cols

        print("T00.t =", T00.t)
        print(
            "dir_rows =", dir_rows, "spacing_rows =", spacing_rows,
            "| dir_cols =", dir_cols, "spacing_cols =", spacing_cols
        )
        print(f"row={row}, col={col}, offset={offset}")
        T_color = SE3(offset[0], offset[1], 0.0) * T00
        print("T_color.t =", T_color.t)
        return T_color

    def _color_palette_get_approach_pose(self, row, col, dist=0.2):
        """Return (T_approach, T_color) for color cell (row, col)."""
        T_color = self._color_palette_get_pose(row, col)
        offset = np.array([0, 0, dist])
        T_approach = SE3(offset) * T_color
        return T_approach, T_color


    def color_palette_gui(self, imgui):
        imgui.separator()
        imgui.text("Color palette calibration")
        imgui.separator()

        # Start-Button
        if self.color_palette_step == 0:
            if imgui.button("Calibrate color palette"):
                self.color_palette_step = 1
                self.color_palette_status = (
                    "Move TCP to color 00 (row 0, col 0) and press 'Set color 00'."
                )

        # Schritt 1: p00
        if self.color_palette_step == 1:
            imgui.text_wrapped("Step 1: Move TCP to color 00 (row 0, col 0).")
            if imgui.button("Set color 00"):
                try:
                    self.color_palette_set_color_00()
                    self.color_palette_step = 2
                    self.color_palette_status = (
                        "Color 00 stored. Move TCP to color 01 (row 0, col 1) and press 'Set color 01'."
                    )
                except Exception as e:
                    self.color_palette_status = f"Error setting color 00: {e}"

        # Schritt 2: p01
        if self.color_palette_step == 2:
            imgui.text_wrapped("Step 2: Move TCP to color 01 (row 0, col 1).")
            if imgui.button("Set color 01"):
                try:
                    self.color_palette_set_color_01()
                    self.color_palette_step = 3
                    self.color_palette_status = (
                        "Color 01 stored. Enter number of colors in one row."
                    )
                except Exception as e:
                    self.color_palette_status = f"Error setting color 01: {e}"

        # Schritt 3: count_rows
        if self.color_palette_step == 3:
            imgui.text_wrapped("Step 3: Enter number of colors in one row.")
            changed, rows = imgui.input_int(
                "Colors in one row", int(self.color_palette.get("count_rows", 3))
            )
            if changed and rows > 0:
                self.color_palette["count_rows"] = int(rows)
                self._save_color_palette()
            if imgui.button("Confirm rows"):
                self.color_palette_step = 4
                self.color_palette_status = (
                    "Rows stored. Move TCP to color 10 (row 1, col 0) and press 'Set color 10'."
                )

        # Schritt 4: p10
        if self.color_palette_step == 4:
            imgui.text_wrapped("Step 4: Move TCP to color 10 (row 1, col 0).")
            if imgui.button("Set color 10"):
                try:
                    self.color_palette_set_color_10()
                    self.color_palette_step = 5
                    self.color_palette_status = (
                        "Color 10 stored. Enter number of rows (colors in column)."
                    )
                except Exception as e:
                    self.color_palette_status = f"Error setting color 10: {e}"

        # Schritt 5: count_cols
        if self.color_palette_step == 5:
            imgui.text_wrapped("Step 5: Enter number of rows (colors per column).")
            changed, cols = imgui.input_int(
                "Colors in one column", int(self.color_palette.get("count_cols, ", 2))
            )
            if changed and cols > 0:
                self.color_palette["count_cols"] = int(cols)
                self._save_color_palette()
            if imgui.button("Finish calibration"):
                self._save_color_palette()
                self.color_palette_step = 0
                self.color_palette_status = "Color palette calibration finished."

        # Status / Info
        if self.color_palette_status:
            imgui.text_wrapped(self.color_palette_status)

        if self.color_palette_is_calibrated():
            imgui.text(
                f"Calibrated: rows={self.color_palette.get('count_rows', '?')}, "
                f"cols={self.color_palette.get('count_cols', '?')}, "
                f"spacing_rows={self.color_palette.get('spacing_in_rows', 0):.3f} m, "
                f"spacing_cols={self.color_palette.get('spacing_in_cols', 0):.3f} m"
            )
        else:
            imgui.text("Calibrated: no")

        if imgui.button("Reset color palette calibration"):
            self.color_palette = {
                "p00": None,
                "p01": None,
                "p10": None,
                "spacing_in_rows": None,
                "direction_rows": None,
                "spacing_in_cols": None,
                "direction_cols": None,
                "count_rows": 3,
                "count_cols": 2,
            }
            self.color_palette_step = 0
            self._save_color_palette()
            self.color_palette_status = "Color palette calibration reset."

        # Row / Col Auswahl
        changed, col = imgui.input_int("Col##color", int(self.color_col))
        if changed and col >= 0:
            self.color_col = col

        changed, row = imgui.input_int("Row##color", int(self.color_row))
        if changed and row >= 0:
            self.color_row = row

        # Script-Pfad
        changed, script = imgui.input_text(
            "Script path##color",
            self.color_script,
            256,
        )
        if changed:
            self.color_script = script


        # Button: get_color aufrufen
        if imgui.button("Get color"):
            try:
                self.get_color(
                    self.color_row,
                    self.color_col,
                    self.color_script,
                )
                self.color_status = (
                    f"get_color(row={self.color_row}, col={self.color_col}, "
                    f"script='{self.color_script}' executed."
                )
            except Exception as e:
                self.color_status = f"Error in get_color: {e}"

        if self.color_status:
            imgui.text_wrapped(self.color_status)

    def read_states(self):
        if self.panda is None:
            # self.robot.q = self.robot.qr
            print("No panda, can't read states")
            return

        self.robot.q = self.panda.get_robot().read_once().q # panda.q
        self.grasped = self.gripper.read_once().is_grasped
        # print('Grasped:', self.grasped)
        
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
        Kp = np.array([600, 600, 600, 600, 250, 150, 50]) #*1.3 #*0.5
        #Kp = np.array([600, 600, 600, 600, 250, 50, 50]) #*0.5
        #Kp = np.array([600, 600, 600, 600, 250, 20, 50]) #*0.5
        #Kp = np.array([1000.0, 1000.0, 1000.0, 1000.0, 500.0, 300.0, 100.0]) #*0.8
        
        Kv = 1.0*2.0*np.sqrt(Kp)
        Kv[-1] = 10.0
        Kv[-2] = 20.0
        Kv[-3] = 20.0
        #print('Kv', Kv)
        # Seems to be more stable
        # Kv = 0.5*np.sqrt(2.0*Kp)
        Kv[-1] = 5.0

        print('Kp', Kp)
        print('Kv', Kv)
        ctrl.set_stiffness(Kp)
        ctrl.set_damping(Kv)
        panda = self.panda
        dq_thresh = 0.001
        dq0 = np.zeros(7)

        self.panda.enable_logging(len(Q)+1000)
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
        self.panda.disable_logging()

    def handeye_image(self):
        img = Image.fromarray(self.handeye.cam_image)
        return img
    
    def _exec_drawing_motion(self, motion):
        from panda_py import controllers
        assert(False)
        Q = motion.Qs
        dQ = motion.dQs

        print('Moving to initial position')
        self.panda.move_to_joint_position(Q[0])

        self.delay(0.5)
        print('Starting Cartesian impedance control loop')
        ctrl = controllers.JointPosition()
        Kp = np.diag([800, 800, 150, 30, 30, 30])

        # NB pandapy uses the Eigen (memory) convention of putting w first in Quaternions
        def eigen_quat(q):
            return [q[1], q[2], q[3], q[0]]

        stiffness_orientation = self.pad.get_pen_orientation(self.robot, Q[0]) #self.robot.q)
        ctrl = controllers.CartesianImpedance(impedance=Kp, filter_coeff=1.0,
                                              nullspace_stiffness=5.0,
                                              stiffness_orientation=eigen_quat(stiffness_orientation))

        # Get trajectory in 'hand space'
        Pos = []
        Orn = []
        for q in Q:
            x = self.robot.hand_tcp_pos(q)
            Pos.append(x[:3])
            Orn.append(eigen_quat(x[3:]))

        n = len(Q)
        i = 0
        dq_thresh = 0.0001

        panda = self.panda
        panda.start_controller(ctrl)
        with panda.create_context(frequency=1000) as ctx:
            while ctx.ok():
                if i < n:
                    ctrl.set_control(Pos[i], Orn[i], Q[i])
                    i += 1
                else:
                    ctrl.set_control(Pos[-1], Orn[-1], Q[-1])
                    dq = panda.get_state().dq
                    i += 1
                    if abs(np.max(dq)) < dq_thresh and i > (n+1000):
                        break
        print('Stopping Cartesian impedance controller')
        self.panda.stop_controller()

    def execute(self, delay=1.0, force_drawing=False):
        ''' Execute the current joint motion'''
        try:
            if self.panda is not None:
                self.delay(delay)

                for motion in self.motions:
                    if motion.grasp is not None:
                        if motion.grasp == False:
                            self.ungrasp()
                        elif type(motion.grasp) == bool:
                            self.grasp()
                        else:
                            self.grasp(motion.grasp)
                        time.sleep(0.1) #1.0)

                    #Q = self.Qs
                    #dQ = self.dQs
                    if motion.is_drawing or force_drawing:
                        self._exec_drawing_motion(motion)
                    else:
                        self._exec_joint_motion(motion)
                return np.array(self.panda.q)
            elif self.has_ros():
                for motion in self.motions:
                    if motion.grasp is not None:
                        if motion.grasp == False:
                            with self.wait():
                                self.send_msg('exec:ungrasp')
                        elif type(motion.grasp) == bool:
                            with self.wait():
                                self.send_msg('exec:grasp')
                            #self.grasp()
                        else:
                            print("Received grasping with parameters, not implemented yet")
                            with self.wait():
                                self.send_msg('exec:grasp')
                            #self.grasp(motion.grasp)
                        #time.sleep(0.3) #1.0)

                    if len(motion.Qs):
                        with self.wait():
                            self.send_joint_trajectory(motion.Qs, motion.dQs)
            else:
                print("No panda or ros, can't execute, simulating")
                self.simulate()
            print('Finished motion')
        except ValueError:
            print("Could not finish motion")

    def goto_joint_position(self, q):
        if type(q) == str:
            name = q
            q = self.editor.get_position(name)
            if q is None:
                print('Invalid position name:', name)
                print(self.editor.positions)
                return
        print('goto ', q)
        if self.panda is not None:
            print("Actually moving the robot")
            self.panda.move_to_joint_position(np.array(q))
            print("Done moving")
        else:
            self.robot.q = q
            if state.has_ros:
                self.send_joint_pos(q)

    def grasp(self, mm=cfg.default_grasp_mm):
        ''' Grasp with a width specified in mm'''
        if self.gripper is None:
            print('No gripper!')
            return
        print('Grasping')
        self.gripper.grasp(mm/1000, 0.2, 10, 0.04, 0.04) # Might want to increase pressure current 10 N

    def ungrasp(self):
        ''' Release the gripper'''
        if self.gripper is None:
            print('No gripper!')
            return
        self.gripper.stop()
        self.gripper.move(100.0/1000, 0.2)

    def go_aruco(self):
        self.update_handeye()
        if self.panda is None and state.has_ros:
            print("Waiting for movement")
            with self.wait(): 
                self.goto_joint_position('Aruco')
        else:
            self.goto_joint_position('Aruco')
        print("Updating handeye")
        self.update_handeye()
        print("Clearing aruco")
        self.clear_aruco()

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

    def clear_motions(self):
        self.motions = []
        self.trajectory_status = True

    def plan_joint_approach(self, q):
        ''' Plan a joint approach to a target position'''
        Qs, dQs = tu.joint_approach(self.robot.q, q)
        self.motions = [Motion(Qs, dQs)]
        self.trajectory_status = self.robot.check_joint_trajectory(Qs, dQs)
        if not self.trajectory_status or self.safe_mode:
            self.plot_trajectory_stats()

    def plan_joint_trajectory(self, Qs, dQs, plot=False, task_trajectories=[]):
        self.read_states()
        # PLOT the trajctory if
        self.trajectory_status = self.robot.check_joint_trajectory(Qs, dQs)
        print('Trajectory status', self.trajectory_status)
        print('Safe mode', self.safe_mode)

        self.motions = [Motion(Qs, dQs)]

        if not self.trajectory_status or self.safe_mode:
            print('Plotting')
            self.plot_trajectory_stats(task_trajectories)

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

    def compute_task_trajectories(self, trajectories, rest=0.0, q=None, stiff_orn=False, tip_frame='tiba_pen_tcp'):
        ''' Plan trajectories for the robot given in task space as
        a sequence of position and quaternion rotation poses [x, y, z, q_0, q_1, q_2, q_3].
        The trajectory is assumed to be sampled at a frequency of (1.0 / cfg.dt) Hz.
        The result is a joint trajectory sampled at (1.0 / cfg.robot_dt) Hz
        '''

        if q is None:
            q = self.robot.q

        precisions = np.ones(7)
        if not stiff_orn:
            precisions[3:] *= self.drawing_orientation_stiffness
        planner = ilqr.iLQRPlannerVelocity(self.robot.urdf_file,
                                           q,
                                           subsample=int(cfg.dt/cfg.robot_dt),
                                           tip_frame=tip_frame
                                           )
        #
        Qs, dQs, linked_trajectories = planner.plan(trajectories,
                                                    rest=rest,
                                                    limits=robot.panda_limits,
                                                    weights=precisions,
                                                    get_trajectories=True,
                                                    r=cfg.ilqr_r)

        self.trajectory_status = self.robot.check_cartesian_trajectories(linked_trajectories)
        if not self.trajectory_status:
            print("Warning in computed cartesian trajectories")
            plot_cartesian_trajectories(linked_trajectories)
            return Qs, dQs
        # planner = ilqr.iLQRPlannerAcceleration(self.robot.urdf_file,
        #                                        q,
        #                                        subsample=int(cfg.dt/cfg.robot_dt),
        #                                        tip_frame=tip_frame
        #                                        )
        # Qs, dQs, linked_trajectories = planner.plan(trajectories, rest=rest, limits=robot.panda_limits, qd_lim=self.robot.qd_lim, get_trajectories=True)

        #plot_cartesian_trajectories(linked_trajectories)
        return Qs, dQs

    def plan_task_trajectories(self, trajectories, rest=0.0, q=None, tip_frame='tiba_pen_tcp'):
        Qs, dQs = self.compute_task_trajectories(trajectories, rest, q, tip_frame)
        self.plan_joint_trajectory(Qs, dQs, trajectories)

    def plan_and_execute_script(self, path, **kwargs):
        if 'Pick' in path and self.grasped:
            print("Cannot execute grip motion")
            return
        self.plan_script(path, **kwargs)
        print(self.trajectory_status)
        if self.trajectory_status:
            self.execute()

    def plan_script_key_name(self, key, append_to_configuration=None):
        ''' Plan a script with a name from the motions.yaml file
        `append_to_configuration` is an optional joint state. If specified,
        current motions will not be cleared
        '''
        if not key in self.scripts:
            print("Script does not exist")
            print(self.scripts.keys())
            return
        script = self.scripts[key]
        args = script['args'] if 'args' in script else {}

        self.plan_script(script['file'],
                         append_to_configuration=append_to_configuration,
                         **args)

    def plan_script(self, path, append_to_configuration=None, **kwargs):
        ''' Plan a script with a given path

        '''
        self.reload_scripts()

        if path in self.scripts:
            return self.plan_script_key_name(path, append_to_configuration)

        # Local functions
        import numpy as np

        path = os.path.join(self.base_path, path)
        
        # Local state to incrementally build trajectory
        state = lambda: None
        if append_to_configuration is not None:
            state.q = append_to_configuration
        else:
            state.q = self.robot.q
        state.pose = self.robot.pen_tcp_pose(state.q)
        state.hand_pose = self.robot.hand_tcp_pose(state.q)

        # Clear joint trajectory or append a new one
        if append_to_configuration is None:
            self.motions = [Motion([], [])]
        else:
            self.motions.append(Motion([], []))

        # self.Qs = []
        # self.dQs = []
        print("Script arguments:")
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
            #print('Updated pose to', state.pose)

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
            if len(Q):
                self.motions[-1].Qs += list(Q)
                self.motions[-1].dQs += list(dQ)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")

        def tool_align(y):
            dest_pose = tu.align_pose_zy(state.pose, [0, 0, -1], [-1, 0, 0])
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")

        def tool_align_dip():
            tool_align([-1, 0, 0])

        def hand_align(y):
            dest_pose = self.robot.get_pos(state.q, 'tiba_hand_tcp')
            dest_pose = tu.align_pose_zy(dest_pose, [0, 0, -1], y)
            #dest_pose = self.robot.hand_frame_to_pen_tcp(dest_pose)
            trajectories = [tu.approach_sequence([state.hand_pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q, tip_frame='tiba_hand_tcp')
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")

        def hand_align_pick_tool():
            hand_align([0, -1, 0])

        def move_xyz(v):
            #print(state.pose)
            dest_pose = np.array(state.pose)
            dest_pose[:3, 3] += v
            #print(dest_pose)
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")

        def goto_xyz_hand(pos):
            #print(state.pose)
            dest_pose = np.array(state.hand_pose)
            dest_pose[:3, 3] = pos
            trajectories = [tu.approach_sequence([state.hand_pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q, tip_frame='tiba_hand_tcp')
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")

        def goto_xyz(pos):
            #print(state.pose)
            dest_pose = np.array(state.pose)
            dest_pose[:3, 3] = pos
            #print(dest_pose)
            trajectories = [tu.approach_sequence([state.pose, dest_pose])]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")
                
        def xy_circles(radius_mm, rep):
            p0 = state.pose.t
            pts = []
            n = 800 # NB that for planar movements TOPP breaks
            radius = radius_mm/1000
            for i in range(rep):
                for a in np.linspace(0, 2*np.pi, n)[:-1]:
                    pts.append(p0 + np.array([np.cos(a), np.sin(a), 0])*radius)
            #pts.append(p0)
            to_plan = [np.array([p0, pts[0]]), np.array(pts), np.array([pts[-1], p0])]
            paths, _, _ = tu.plan_polylines_time_optimal( to_plan,
                                                                  max_speed=cfg.cartesian_speed,
                                                                  accel_time=cfg.acceleration_time,
                                                                  tol=0.01)
            orn = tu.to_pos_orn(state.pose)[3:]
            trajectories = [[np.concatenate([p, orn]) for p in P] for P in paths]
            Qs, dQs = self.compute_task_trajectories(trajectories, q=state.q)
            if len(Qs):
                self.motions[-1].Qs += list(Qs)
                self.motions[-1].dQs += list(dQs)
                update_state(self.motions[-1].Qs[-1])
            else:
                print("Empty trajectory")
            
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

        self.trajectory_status = self.robot.check_joint_trajectory(self.Qs, self.dQs)
        if append_to_configuration is None:
            if not self.trajectory_status or self.safe_mode:
                print("Plotting with safe mode:", self.safe_mode, "status:", self.trajectory_status)
                self.plot_trajectory_stats()

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

