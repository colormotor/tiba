from py5canvas import *

import numpy as np
import cv2

# import yarp
import time
from easydict import EasyDict as edict
from slimgui import imgui
from collections import defaultdict
from tiba import svg, spatial, interface, config, robot, ui, planning, geom, plut
import os
from importlib import reload

reload(planning)

from scipy.ndimage import gaussian_filter1d
import tiba.trajectory_utils as tu

# ROS Setup
#
os.environ["ROS_IPV6"] = "off"
# os.environ['ROS_MASTER_URI']='http://192.168.1.227:11311' #Running ROS
# os.environ['ROS_MASTER_URI']='http://172.16.16.38:11311' #Running ROS
#
# os.environ['ROS_MASTER_URI']='http://10.1.1.5:11311' #Running ROS
#

# os.environ['ROS_MASTER_URI']='http:/172.16.17.243:11311' #Running ROS
os.environ["ROS_MASTER_URI"] = "http://172.16.16.185:11311"  # Running ROS
os.environ["ROS_MASTER_URI"] = "http://192.168.8.200:11311"  # Running ROS

interface.state.has_ros = (
    False  # Set to false if you don't need to communicate with the server
)

base_path = "./../robots/tiba/settings"  # tiba for London
itiba = interface.TibaInterface(
    init_node="tiba_notebook",
    base_path=base_path,
    # drawing_orientation_stiffness=0.01) #0.00001)
    drawing_orientation_stiffness=0.00001,
)
# drawing_orientation_stiffness=0.1)
itiba.set_tool(itiba.tools["Probe"])
itiba.safe_mode = True
cfg = config.cfg

# cfg.ilqr_r = 1e-6 #5 #1e-4 #1e-5
# cfg.speed_factor = 1.0 #0.9
# cfg.acceleration_time = 0.3 #0.4 # 0.2
# cfg._default_joint_speed = geom.radians(70) #80)

itiba.print_available_tools()
itiba.print_available_positions()
itiba.print_available_scripts()
itiba.send_msg("getq")
# %%

VISUAL_FEED = False

state = lambda: None
state.planned = False
state.plan_image = None
state.test_path = None
state.trajectory_type = None
state.trajectories = []
state.paths = []
state.viz_trajectories = []
state.topp_trajs = []
state.draw_next = False
state.server = None

tag = "sorting_demo_slm_tag"


def brush_name():
    # return 'Mol'
    # return 'Molotow-brush'
    # return 'HK-Calli-med-02'
    # return 'Probe'
    return "Chisel-brush-manifold-01"
    return "Thin-manifold-01"
    return "Labor-gold-0_0"
    return "Spongie"
    return "HK-Calli-small-demo"

    return "HK-Calli-thin-01"
    return "Spring-whsmith"
    return "Spring-pencil"
    return "HK-Calli-med-01"

    # return 'Calli-thin-01'
    # return 'Chisel-02'
    # return 'Labor-brush-ipai-4_0'

    return "Fude"


def dip_command():
    ind = 0
    return "Dip-%d" % ind
    return "Dip-rim-%d" % ind


gen_size = 512
robot_size_mm = 250
cfg.ilqr_r = 1e-9  # 1e-6
cfg.speed_factor = 1.0  # 0.7 #1.0 #1.0 #0.9
cfg.acceleration_time = 0.3
cfg._default_joint_speed = geom.radians(40)

# gfxui.cfg.dragger_size = np.array([9, 9])

if VISUAL_FEED:
    video = VideoInput(1)  # , size=(512, 512))
    frame = video.read()
    video_w, video_h = frame.size
    print("video size", video_w, video_h)
else:
    video_w, video_h = 900, 600  # 512, 512

camera_rect = geom.make_rect(0, 0, video_w, video_h)
source_rect = geom.make_centered_rect(
    geom.rect_center(camera_rect), [gen_size, gen_size]
)


class VisualFeedback:
    def __init__(self, src_rect):
        self.positions = np.array(geom.rect_corners(src_rect))
        self.dst_points = np.array(geom.rect_corners(src_rect))

    def load(self, path):
        if os.path.isfile(path):
            rec = files.load_json(path)
            self.positions = np.array(rec["positions"])

    def save(self, path):
        rec = {"positions": self.positions}
        files.save_json(rec, path)

    def gui(self, sketch, image):
        c = sketch.canvas
        mat = geom.scaling_2d(0.23)
        inv_mat = np.linalg.inv(mat)
        pts = geom.tsm(mat, self.positions)

        gfxui.begin(sketch)
        # Only interact (updates points)
        for i, p in enumerate(pts):
            pts[i] = gfxui.dragger(i, p)
        # c.circle(sketch.mouse_pos, 15)

        self.positions = geom.tsm(inv_mat, pts)

        # UI works here
        c.push_matrix()
        c.apply_matrix(mat)
        c.image(image, 0, 0)  # , width, height)
        c.pop_matrix()

        c.stroke(255)
        c.no_fill()
        c.polygon(pts)
        gfxui.end()

    def warped(self, image):
        im_warped = warp_image_to_rectangle(
            image, self.positions, dst_points=self.dst_points
        )
        return im_warped


if VISUAL_FEED:
    vfeed = VisualFeedback(source_rect)


def width_profile_speed(s, minw, maxw):
    """speed-based width profile"""
    w = np.exp(-(s + 1) / 1)  # (np.mean(s)+1e-100))
    return minw + (maxw - minw) * w


def normalize_speed(s):
    return (s - s.min()) / (s.max() - s.min())


def split_width(P, threshold):
    if type(P) == list:
        return sum([split_width(p, threshold) for p in P], [])

    if P.shape[1] != 3:
        raise ValueError("Input polyline must be an Nx3 numpy array.")
    segments = []
    current_segment = []
    for point in P:
        x, y, w = point
        if w >= threshold:
            current_segment.append([x, y, w])
        else:
            if len(current_segment) >= 2:
                segments.append(np.array(current_segment))
            current_segment = []
    if len(current_segment) >= 2:
        segments.append(np.array(current_segment))
    return segments


def trajectories_from_scene(path, delta_t_mul=1.0, subd=15, preview=False):
    dt = cfg.dt
    strokes = files.load_json(path)

    trajectories = []
    speeds = []
    paths = []
    src_rect = geom.make_rect(0, 0, gen_size, gen_size)
    dst_rect = geom.scale_rect(src_rect, params.scale)
    mat = geom.rect_in_rect_transform(src_rect, dst_rect)
    mat = geom.trans_2d([params.px, params.py]) @ mat
    scale = np.sqrt(np.linalg.det(mat[:2,:2]))
    max_speed = cfg.cartesian_speed
    for group in strokes["groups"]:
        for shape in group["shapes"]:
            m = len(shape["points"]) - 1
            P = geom.tsm(mat, np.array(shape["points"]))
            stroke_w = np.array(shape["stroke_width"]) * scale
            # P = geom.tsm(mat, P)
            Delta_t = np.array(shape["Delta_t"])
            if geom.is_number(stroke_w):
                stroke_w = np.ones(len(P)) * stroke_w

            path = dslm.SLMTrajectory(
                points=P,
                stroke_width=stroke_w,
                Delta_t=Delta_t * delta_t_mul,  # shape['Delta_t'],
                Ac=(shape["Ac"], False),
                subd=5,
                delta=(shape["delta"], False),
            )
            endt = path.endt.detach().cpu()
            n = int(endt / dt)
            scale_factor = gen_size / (robot_size_mm / 1000)
            print("Local scale factor", 1 / scale_factor)
            max_speed = (400 / 1000) * scale_factor
            # max_speed = cfg.cartesian_speed*scale_factor
            if preview:
                print("Preview")
                X = path.samples(1).detach().cpu().numpy()
                print("Result", len(X))
            else:
                X = (
                    path.samples_with_bounds(
                        max_speed, cfg.acceleration_time * 1, dt=dt
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            dX = np.diff(X, axis=0) / dt
            speed = np.max(np.linalg.norm(dX, axis=1))
            print(
                "Maxspeed: ",
                cfg.cartesian_speed,
                speed / scale_factor,
                speed * scale_factor,
            )
            # X = path.bezier_samples(10*len(path.get_points()))
            # X = path.get_points().detach().cpu().numpy()

            # samples, dX, ssss = path.samples(n, get_strokes=True, velocity=True)
            # dX = dX.detach().cpu().numpy()
            # X = X.detach().cpu().numpy()
            # if X.shape[1] < 3:
            #     dX = np.diff(X, axis=0)
            #     speed =  np.sqrt(dX[:,0]**2 + dX[:,1]**2)
            #     speed = np.concatenate([speed[:1], speed])

            #     speed = normalize_speed(speed)

            #     w = width_profile_speed(speed, 0.5, 10.0)
            #     w = gaussian_filter1d(w, 2)
            #     X = np.vstack([X.T, w]).T
            # speeds.append(np.sqrt(dX[:,0]**2  + dX[:,1]**2))
            paths.append(path)
            trajectories.append(X)  # [::-1])

    return trajectories, paths


def parameters():
    return dict(
        alpha=(0.0, {"min": 0.0, "max": 1.0}),
        scale=(1.0, {"min": 0.1, "max": 2.5}),
        px=(1.0, {"min": -700, "max": 700}),
        py=(1.0, {"min": -700, "max": 700}),
        spline_subd=(5, {"type": "int", "min": 1, "max": 50}),
        dip_distance=(1.0, {"min": 0.1, "max": 10.0}),
        dt_mul=(1.0, {"min": 0.4, "max": 2.0}),
        plane_offset_mm=(0.0, {"min": 0.0, "max": 15.0}),
        topp=False,
        split_w=False,
        calibrate_camera=False,
        Testing=dict(
            calibrate_viz=False,
            no_dip=False,
            active=False,
            pos_x=(10.0, {"min": 0.0, "max": gen_size}),
            pos_y=(10.0, {"min": 0.0, "max": gen_size}),
            length=(100.0, {"min": 10.0, "max": gen_size}),
            plot_rect=False,
        ),
    )


def gui():
    ui.imgui_text_separator("Viz")
    ui.imgui_text_separator("Robot")

    # if ui.imgui_color_button('TOPP'):
    #     scale_factor = gen_size/(robot_size_mm/1000)
    #     max_speed = cfg.cartesian_speed*scale_factor
    #     S, _, _ = tu.plan_polylines_time_optimal([P[:,:2] for P in state.trajectories], max_speed, cfg.acceleration_time, tol=10, ds=2.5) #/1000.0):
    #     state.topp_trajs = [P[:,:2] for P in S]
    #     print('SLM trajectory duration', len(state.trajectories[0])*cfg.dt)
    #     print('TOPP trajectory duration', len(state.topp_trajs[0])*cfg.dt)

    if ui.imgui_color_button("Home"):
        itiba.goto_joint_position("Home")
    if ui.imgui_color_button("Plan"):
        plan_current()

    if state.planned:
        imgui.same_line()
        btn = "Execute"
        if not interface.state.has_ros:
            btn = "Simulate"
        if ui.imgui_color_button(btn):
            print("Executing")
            if interface.state.has_ros:
                itiba.execute_proc()
            else:
                itiba.simulate()
            state.planned = False

    imgui.same_line()
    if ui.imgui_color_button("Dip"):
        state.planned = False
        itiba.plan_script(dip_command())
        if itiba.trajectory_status:
            state.planned = True

    if imgui.button("Clear"):
        state.trajectories = []
        state.viz_trajectories = []


def plan_and_execute():
    plan_current()

    if interface.state.has_ros:
        itiba.execute()
    else:
        itiba.simulate()


def plan_current():
    itiba.safe_mode = True
    itiba.set_tool(brush_name())

    if state.trajectory_type == "bspline":
        planner = "Calligraph"
    elif state.trajectory_type == "poly":
        planner = "Trapezoidal"
    elif state.trajectory_type == "slm":
        planner = "Pass"
    else:
        raise ValueError("Unknown trajectory type", state.trajectory_type)

    if state.trajectory_type == "slm" and params.topp:
        planner = "Trapezoidal"

    planner = planning.TrajectoryPlanner(
        itiba,
        planner=planner,
        resample_topp=params.topp,
        topp_tol_mm=1,  # 10,
        plane_offset_mm=params.plane_offset_mm,
    )

    paths_to_plan = []
    if params.testing.active:
        paths_to_plan = [state.test_path]
    else:
        paths_to_plan = state.trajectories

    # Fit to destination rectangle
    dest_rect = planning.make_rect_mm(0, 0, robot_size_mm, robot_size_mm)
    src_rect = geom.make_rect(0, 0, gen_size, gen_size)
    planner.draft = paths_to_plan
    planner.fit_draft(dest_rect, src_rect=src_rect)

    # Log maximum speed
    for X in planner.trajectories:
        dX = np.diff(X, axis=1)[:, :3] / cfg.dt
        speed = np.max(np.linalg.norm(dX, axis=1))
        print("MAXSPEED", speed)

    # Dip logic
    dip_distance = params.dip_distance
    predip = True
    if params.testing.active or params.testing.no_dip:
        predip = False
        dip_distance = 10000
    planner.compute_and_plan_with_dips(
        predip=predip, distance=dip_distance, dip_command=dip_command()
    )

    state.planned = False
    state.plan_image = None
    if itiba.trajectory_status:
        state.planned = True
        state.plan_image = planner.plot_plan_image(backend="image", show=True)


# def plan_polylines(S):
#     state.trajectories = []
#     for X in S:
#         X = geom.uniform_sample(X, 1)
#         X = np.hstack([X, np.ones((len(X),1))])
#         state.trajectories.append(X) #Y)
#         #
#     itiba.safe_mode = True
#     itiba.set_tool(brush_name())
#     planner = 'Trapezoidal'

#     planner = planning.TrajectoryPlanner(itiba, planner=planner,
#                                          resample_topp=True,
#                                          topp_tol_mm=5, #2, #10,
#                                          plane_offset_mm=params.plane_offset_mm
#                                          )

#     paths_to_plan = []
#     if params.testing.active:
#         paths_to_plan = [state.test_path]
#     else:
#         paths_to_plan = state.trajectories


#     #paths_to_plan = paths_to_plan[:2]
#     dest_rect = planning.make_rect_mm(0, 0, robot_size_mm, robot_size_mm)
#     src_rect = geom.make_rect(0, 0, gen_size, gen_size)
#     planner.draft = paths_to_plan
#     planner.fit_draft(dest_rect, src_rect=src_rect)

#     for X in planner.trajectories:
#         dX = np.diff(X, axis=1)[:,:3]/cfg.dt
#         speed = np.max(np.linalg.norm(dX, axis=1))
#         print('MAXSPEED', speed)

#     dip_distance = params.dip_distance #dip_distances[params.drawing_layer][params.drawing_pass]
#     predip = True
#     if params.testing.active or params.testing.no_dip:
#         predip = False
#         dip_distance = 10000
#     planner.compute_and_plan_with_dips(predip=predip,
#                                        distance=dip_distance,
#                                        dip_command=dip_command())
#     state.planned = False
#     state.plan_image = None
#     if itiba.trajectory_status:
#         state.planned = True
#         state.plan_image = planner.plot_plan_image(backend='image', show=True)


def split_trajectories(paths):
    d = 0.0
    chunk_scale = gen_size / (0.001 * robot_size_mm)
    distance = (
        params.dip_distance
    )  # [params.drawing_layer][params.drawing_pass]*chunk_scale
    # traj_dist = np.sum([geom.chord_length(traj) for traj in paths])
    # num_chunks = max(1, traj_dist/distance)
    # ad
    # distance = traj_dist/num_chunks

    chunks = [[]]
    for traj in paths:
        chunks[-1].append(traj)
        d += geom.chord_length(traj)
        if d > distance:
            chunks.append([])
            d = 0
    return [chunk for chunk in chunks if chunk]


def transform_paths(mat, paths):
    s = np.sqrt(np.linalg.det(mat[:2,:2]))
    res = []
    for P in paths:
        P = np.array(P)
        P[:, :2] = geom.tsm(mat, P[:, :2])
        if P.shape[1] > 2:
            P[:, 2] *= s
        res.append(P)
    return res


def load_file():
    c = sketch.canvas
    # name = 'demo_slm_tag_04_CREAM__024.json'
    # name = 'demo_slm_tag_04_CREAM_s1.2_029.json'
    # name = 'demo_slm_tag_04_CREAM_s1.2_019.json'
    # name = 'demo_slm_tag_04_CREAM_s1.2_006.json'
    # fl = os.path.join(base_path, name)
    # fl = paths[params.tag_index]
    # fl = os.path.join(base_path, 'demo_slm_tag_04_CREAM_s1.2_019.json')
    # fl = '/Users/colormotor/Dropbox/transfer_box/data/roman/reconstructions/demo_slm_gml_01_037.json'
    fl = os.path.expanduser(
        "~/Dropbox/transfer_box/data/anna/outputs/demo_slm_anna_02_099.json"
    )
    fl = os.path.expanduser(
        "~/Dropbox/transfer_box/data/calligraph/outputs/demo_slm_gml_01_045.json"
    )
    fl = os.path.expanduser(
        "~/Dropbox/transfer_box/data/calligraph/outputs/demo_slm_portrait_01_116.json"
    )
    # fl = os.path.expanduser('~/Dropbox/transfer_box/data/roman/portraits/demo_slm_portrait_01_135.json')
    # fl = os.path.expanduser('~/Dropbox/transfer_box/data/calligraph/outputs/demo_asia_areafill_01_acal4_n100_s3000_bsp1_lr2.0_mse20.0_001.json')
    name = "demo_asia_areafill_01_acal3_n300_s100_bsp1_lr1.5_mse10.0_001"
    name = "demo_asia_iterative_stroke_01__sds_053"
    name = "demo_asia_iterative_stroke_01__sds_099"
    name = "demo_asia_iterative_stroke_01__sds_100"
    name = "demo_asia_iterative_stroke_01__sds_101"
    name = "demo_slm_portrait_01_135"
    name = "demo_area_fill_01_G_abl-x_001"
    name = "demo_slm_study_portraits__sds_033"
    name = "demo_slm_study_portraits__sds_032"
    name = "demo_slm_study_portraits__sds_036"
    # name = 'demo_slm_study_portraits__sds_070'
    # name = 'demo_slm_study_portraits__sds_074'
    # name = 'demo_slm_study_portraits__sdsLM-226_004'
    # name = 'demo_slm_study_portraits__sdsLM-226_005'
    # name = 'demo_slm_study_portraits__sdsAF-220_002'

    name = "demo_slm_study_portraits__sdsAM-217_002"
    name = "demo_asia_iterative_stroke_01__sds_108"
    name = "demo_asia_iterative_stroke_01__sds_099"

    # name = 'demo_slm_study_portraits__sdsLF-241_003'
    # name = 'demo_slm_study_portraits__sdsBF-039_001'

    # name = 'demo_bspline_different_inits_f_stl200.0_p4_s100.0_t0.5_sds_tsp_001'
    # name = 'demo_bspline_different_inits__clip_hilbert_001'

    # name = "repro_area_81_A_abl-x_001"
    # #name = 'demo_asia_areafill_01_acal3_n150_s10_bsp1_lr1.5_mse10.0_001'
    # #name = 'demo_asia_areafill_01_zcal2_s0_bsp1_lr1.0_abl-x_001'
    # name = "demo_asia_areafill_01_G_abl-x_047"  # Write
    # name = "demo_asia_areafill_01_G_abl-x_045"  # Calli
    # name = "demo_asia_areafill_01_G_abl-x_055"
    # name = 'demo_slm_study_portraits__sds_002'
    # name = 'demo_slm_study_portraits__sds_006'
    # name = 'demo_slm_study_portraits__sds_007'
    # name = 'demo_slm_study_portraits__sds_021'
    # name = 'demo_slm_study_portraits__sds_022'
    # name = 'demo_slm_study_portraits__sds_025'

    # name = 'demo_asia_iterative_dog_steps__sds_001'
    # name = 'demo_asia_areafill_01_acal3_n150_s0_bsp0_lr1.5_mse5.0_001'
    # name = 'demo_asia_areafill_01_acal3_n150_s0_bsp1_lr2.0_mse10.0_001'
    # name = 'demo_asia_areafill_01_G_abl-x_046'

    # name = 'demo_asia_areafill_01_R_abl-x_007'

    # name = 'demo_asia_iterative_stroke_01_bach1_CLIPAG_music2_200.0_s600.0_new_sds_001'
    # name = 'demo_asia_areafill_01_music2_n300_s10_bsp1_lr1.5_mse20.0_001'
    # name = 'demo_bspline_strokes_mediapipe_01__sds_020'
    # name = 'demo_bspline_different_inits__sds_001'
    # name = 'demo_asia_areafill_comp_M_d5_s30.0_007'
    # name = 'demo_asia_areafill_comp_M_d5_s100.0_001'
    # name = 'demo_asia_areafill_comp_M_d5_s10.0_007'
    # name = 'demo_bspline_different_inits_f_stl0.0_p3_s100000.0_t0.5_sds_tsp_001'

    # name = "demo_slm_gml_01_048"
    # name = 'demo_asia_iterative_stroke_01_bach1_CLIPAG_music2_200.0_s600.0_new_sds_001'
    # fl = os.path.expanduser(f'~/Dropbox/transfer_box/data/calligraph/outputs/{name}.json')
    # name = 'ben_demo_asia_areafill_01_zcal2_n150_s500_bsp1_lr1.5_mse10.0_abl-x_001'
    # name = 'ben_demo_asia_areafill_01_acal5_n150_s500_bsp1_lr1.5_mse10.0_abl-x_001'
    fl = os.path.expanduser(
        f"~/Dropbox/transfer_box/data/calligraph/outputs/{name}.json"
    )
    fl = '/home/danielsp/Downloads/camel_16strokes_seed0_best.svg'
    # name = "bez5"
    # fl = os.path.expanduser(
    #     f"~/Dropbox/develop_box/py5sketches/drawlab_vectorize/{name}.bez"
    # )

    # fl = os.path.expanduser(f"~/Dropbox/images/manifold3/manifold2.svg")
    state.file = fl

    if "slm" in fl:
        trajectories, _ = trajectories_from_scene(fl)
        state.trajectories = trajectories
        state.trajectory_type = "slm"
    elif ".svg" in fl:
        S = svg.load_svg(fl)
        src_rect = geom.bounding_box(S)
        dst_rect = geom.scale_rect(
            geom.make_rect(0, 0, gen_size, gen_size), params.scale
        )
        mat = geom.rect_in_rect_transform(src_rect, dst_rect)
        mat = geom.trans_2d([params.px, params.py]) @ mat
        S = geom.tsm(mat, S)
        # S = geom.transform_to_rect(S, geom.make_rect(0, 0, gen_size, gen_size))
        state.trajectories = [np.column_stack([P, np.ones(len(P))]) for P in S]
        state.trajectory_type = "poly"
    else:
        src_rect = geom.make_rect(0, 0, gen_size, gen_size)
        dst_rect = geom.scale_rect(src_rect, params.scale)
        mat = geom.rect_in_rect_transform(src_rect, dst_rect)
        mat = geom.trans_2d([params.px, params.py]) @ mat
        scale = np.sqrt(np.linalg.det(mat[:2,:2]))
        use_bspline = not params.topp
        if use_bspline:
            trajectories = diffvg_utils.load_stroke_control_points(
                fl, default_cls="DynamicBSpline"
            )
        else:
            trajectories = diffvg_utils.sample_strokes(fl, default_cls="DynamicBSpline")

        init_box = geom.bounding_box([P[:, :2] for P in trajectories])
        print(init_box)
        m = geom.rect_in_rect_transform(init_box, src_rect)
        trajectories = transform_paths(m, trajectories)

        trajs = []

        for P in trajectories:
            P = np.array(P)
            P[:, :2] = geom.tsm(mat, P[:, :2])

            if P.shape[1] < 3:
                P = np.hstack([P, np.ones((len(P), 1))])
            P[:, 2] *= scale
            trajs.append(P)

        state.trajectory_type = "bspline"
        if use_bspline:
            state.trajectories = trajs
        else:
            state.trajectories = []  # split_width(trajs, 0.1)
            for traj in split_width(trajs, 0.25):
                parts = [Q for Q in split_by_angle(traj, 150) if len(Q) > 1]
                state.trajectories += parts
            state.trajectory_type = "poly"
    # state.trajectories = split_width(state.trajectories, 0.1)
    state.viz_trajectories = []

    for X in state.trajectories:
        S = geom.thick_curve(X) #, add_cap=False)
        state.viz_trajectories.append(S)


# def split_polyline_by_angle(polyline, angle_thresh_deg):
#     """
#     Splits a polyline at points where the angle between segments exceeds the threshold.

#     Parameters:
#     - polyline: (n x 2) numpy array of coordinates
#     - angle_thresh_deg: angle threshold in degrees

#     Returns:
#     - List of polylines (each a numpy array)
#     """
#     ang_thresh = geom.radians(angle_thresh_deg)

#     if len(polyline) < 3:
#         return [polyline]

#     segments = []
#     start_idx = 0

#     for i in range(1, len(polyline) - 1):
#         v1 = polyline[i] - polyline[i - 1]
#         v2 = polyline[i + 1] - polyline[i]
#         angle = abs(geom.angle_between(v1[:2], v2[:2]))

#         if angle > ang_thresh:
#             segments.append(polyline[start_idx:i+1])
#             start_idx = i

#     segments.append(polyline[start_idx:])
#     return segments


def split_by_angle(P, ang_deg):
    ang = geom.radians(ang_deg)
    angles = np.abs(np.concatenate([[0], geom.turning_angles(P[:, :2]), [0]]))
    I = list(np.where(angles > ang)[0])
    if not I:
        return [P]
    I = [0] + I + [len(P) - 1]
    return [P[a : b + 1] for a, b in zip(I, I[1:])]


def add_polylines(S):
    c = sketch.canvas

    state.file = None
    # state.trajectories = []
    # for P in S:
    #     parts = [Q for Q in split_polyline_by_angle(P, 170) if len(Q) > 1]
    # for X in S:
    #     X = geom.uniform_sample(X, 1)
    #     X = np.hstack([X, np.ones((len(X),1))])
    #     state.trajectories.append(X) #Y)

    # TOPP has trouble with sharp turns, so we split trajectories here
    for P in S:
        parts = [Q for Q in split_by_angle(P, 170) if len(Q) > 1]
        for X in parts:
            X = geom.uniform_sample(X, 1)
            X = np.hstack([X, np.ones((len(X), 1))])
            state.trajectories.append(X)  # Y)

    state.viz_trajectories = []

    for X in state.trajectories:
        S = geom.thick_curve(X, add_cap=False)
        state.viz_trajectories.append(S)


def clear():
    state.planned = False
    state.plan_image = None
    state.trajectories = []
    state.viz_trajectories = []
    state.topp_trajs = []


def setup():
    create_canvas(900, 600, 400)
    color_mode("RGB", 1.0)
    itiba.set_tool(brush_name())
    load_file()
    if VISUAL_FEED:
        vfeed.load("vfeed.json")

    title("tiba sender")


def exit():
    if VISUAL_FEED:
        vfeed.save("vfeed.json")


def tick_rect(r, n=25, ts=10):
    stroke(0.5)
    rect(r)
    w, h = geom.rect_size(r)
    y = r[1][1]
    x0 = r[0][0]
    xs = np.linspace(0, w, n)
    # print(xs)
    stroke(0.5, 0.5)
    for x in xs:
        line(x0 + x, y, x0 + x, y + ts)


def draw():
    global colors

    if state.draw_next:
        plan_and_execute()
        clear()
        state.draw_next = False

    must_send_img = False
    # stuff = yarpie.listen()
    # if stuff:
    #     if 'END' in stuff:
    #         print("RECV END")
    #         state.draw_next = True
    #     elif 'get_image' in stuff:
    #         print("Will send image")
    #         must_send_img = True
    #     else:
    #         state.draw_next = False
    #         stuff = stuff.replace('freeline ', '')

    #         pts = eval(eval(stuff))
    #         #pts = [(gen_size-p[1], gen_size-p[0]) for p in pts]
    #         add_polylines([np.array(pts)])

    background(1.0)
    no_stroke()

    push_matrix()
    translate(0, 100)
    scale(0.75)

    if VISUAL_FEED:
        frame = video.read().convert("L")
        img = vfeed.warped(frame)
        x, y = source_rect[0].astype(int)
        feedback_img = img[y : y + gen_size, x : x + gen_size]

        image(img)
        fill(1, 0.5)
        no_stroke()
        rectangle(0, 0, img.shape[1], img.shape[0])

    translate(source_rect[0])

    if params.testing.calibrate_viz:
        image(feedback_img, 0, 0)

    stroke(0.0, 0.5)  # 0,1,0)
    no_fill()
    # rect(0, 0, gen_size, gen_size)
    tick_rect(geom.make_rect(0, 0, gen_size, gen_size), 25)

    # Hack
    ratio_w = 443 / robot_size_mm
    ratio_h = 520 / robot_size_mm
    rrr = geom.make_centered_rect(
        [gen_size / 2, gen_size / 2], [gen_size * ratio_w, gen_size * ratio_h]
    )
    stroke(0.0, 0.15)  # 0,1,0)
    rect(rrr)

    no_stroke()
    fill(0.7)
    text(f"{robot_size_mm} X {robot_size_mm} mm", gen_size + 10, gen_size)

    for i, S in enumerate(state.viz_trajectories):
        fill(plut.default_color(i))
        shape(S)

    if state.topp_trajs:
        stroke(1, 0, 0)
        no_fill()
        shape(state.topp_trajs)

    if params.testing.active:
        no_stroke()
        fill(0, 0, 0, 0.75)
        rectangle(0, 0, gen_size, gen_size)

        if params.testing.plot_rect:
            P = geom.close(geom.shapes.rectangle(0, 0, gen_size, gen_size))
            Pw = np.hstack([P, np.zeros((len(P), 1))])
            state.test_path = Pw
        else:
            # Max width
            maxw = 6  # 0
            # for layer, paths in layer_paths.items():
            #     if layer == params.drawing_layer:
            #         chunks = split_trajectories(paths)
            #         num_chunks = len(chunks)
            #         for i, chunk in enumerate(chunks):
            #             for path in chunk:
            #                 maxw = max(maxw, np.max(path[:,2]))

            num_steps = int(params.testing.length * np.pi * 2) * 2

            t = np.linspace(0, np.pi * 2, num_steps)[::-1]

            x = np.cos(t) * params.testing.length + params.testing.pos_x
            y = np.sin(t) * params.testing.length + params.testing.pos_y
            Q = np.vstack([x, y]).T
            np.random.seed(123)
            # Q = geom.shapes.random_radial_polygon(9, 0.1, 1.5)*params.testing.length + [params.testing.pos_x,
            #                                                                  params.testing.pos_y]
            # Q = geom.shapes.star(1, n=5)*params.testing.length + [params.testing.pos_x,
            #                                                                  params.testing.pos_y]
            Q = geom.uniform_sample(Q, 0.5, closed=True)
            Q = geom.close(Q)
            num_steps = len(Q)
            widths = np.linspace(0.0, maxw, num_steps)
            # widths = np.linspace(2, 2, num_steps)

            # x = np.linspace(params.testing.pos_x, params.testing.pos_x+params.testing.length, num_steps)
            # y = np.linspace(params.testing.pos_y, params.testing.pos_y+params.testing.length, num_steps)
            # Plot original widths
            path = np.column_stack([Q, widths])
            state.test_path = path
            S = geom.thick_curve(path, add_cap=False)
            fill(0, 1, 1, 0.7)
            shape(S)

            # Convert to mm
            ratio = robot_size_mm / gen_size
            widths_m = widths * ratio * 0.001
            # xprint(widths_m * 1000)
            z = planning.radius_to_z(itiba, widths_m)
            widths_m_adjusted = planning.z_to_radius(itiba, z)
            widths_px_adjusted = (widths_m_adjusted * 1000) / ratio
            path = np.column_stack([Q, widths_px_adjusted])
            S = geom.thick_curve(path, add_cap=False)
            fill(1, 0.0, 0.0, 0.9)
            shape(S)

    pop_matrix()

    # Cp = state.trajectories[0]
    # subd = 30
    # X = bezier.bezier_piecewise(Cp, subd, d=0)
    # dX = bezier.bezier_piecewise(Cp, subd, d=1)
    # speed = np.linalg.norm(dX, axis=1)

    # no_fill()
    # stroke(1, 0, 0)
    # begin_contour()
    # for i, x in enumerate(np.linspace(0, width-10, len(speed))):
    #     vertex(x, height-speed[i]*100)
    # end_contour()
    # INFO
    text_size(15)
    no_stroke()
    fill(1)
    y = 30
    x = 5
    inc = 25
    fill(0)
    text("Brush: %s" % (brush_name()), [x, y])
    y += inc
    text("Dip: %s" % (dip_command()), [x, y])
    y += inc
    # text('Num chunks: %d'%(num_chunks), [x, y]); y += inc
    if params.testing.no_dip:
        fill(1, 0, 0)
        text("NO DIP!", [x, y])
    fill(1, 0, 0)

    if (
        "scale" in sketch.gui.changed
        or "dt_mul" in sketch.gui.changed
        or "tag_index" in sketch.gui.changed
        or "px" in sketch.gui.changed
        or "py" in sketch.gui.changed
    ):
        load_file()

    if params.testing.calibrate_viz and VISUAL_FEED:
        vfeed.gui(sketch, frame)

    if state.plan_image is not None:
        ref_w = 200
        w, h = state.plan_image.size
        ref_h = ref_w * (h / w)
        image(state.plan_image, width - ref_w, 0, ref_w, ref_h)


def warp_image_to_rectangle(
    image, src_points, dst_size=None, dst_points=None, flip="none"
):
    if not isinstance(image, np.ndarray):
        image = np.array(image) / 255
    src_points = np.array(src_points, dtype=np.float32)
    if dst_size is None:
        dst_size = image.shape[1], image.shape[0]
    width, height = dst_size
    if dst_points is None:
        dst_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

    h_matrix, _ = cv2.findHomography(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, h_matrix, (width, height))
    if flip == "horizontal":
        warped_image = cv2.flip(warped_image, 1)
    elif flip == "vertical":
        warped_image = cv2.flip(warped_image, 0)
    elif flip == "both":
        warped_image = cv2.flip(warped_image, -1)
    return warped_image


run(show_toolbar=True)
