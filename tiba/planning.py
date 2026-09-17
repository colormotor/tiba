from importlib import reload
from . import config, geom, utils, bezier, svg, spatial
from . import trajectory_utils as tu
from . import matplotlyb as ply
from .interface import Motion

from easydict import EasyDict as edict
from PIL import Image
import numpy as np

import pdb

cfg = config.cfg
log = config.log


def make_rect_mm(*args):
    if len(args) == 4:
        c, s = args[:2], args[2:4]
    if len(args) == 3:
        c, s = args[0], args[1:3]
    if len(args) == 2:
        c, s = args[0], args[1]
    return geom.make_centered_rect(np.array(c) / 1000, np.array(s) / 1000)


def plan_paths(itiba, paths, dest_rect, src_rect=None, planner="Trapezoidal", **kwargs):
    pl = TrajectoryPlanner(itiba, planner=planner, **kwargs)
    pl.draft = [np.array(P) for P in paths]
    pl.fit_draft(dest_rect, src_rect=src_rect)
    return pl


class PlannedTrjaectories:
    def __init__(
        self,
        planner,
        draft,
        dest_rect=None,
        max_paths=None,
        select_indices=[],
        src_rect=None,
    ):
        pad = planner.itiba.pad
        pad.offset = (
            planner.plane_offset_mm / 1000
        )  # + self.itiba.tool_info.offset_mm / 1000
        pad.pos_offset = np.array(planner.offset_mm) / 1000
        pad.update()
        if dest_rect is not None:
            pad.scaled_box = dest_rect
        paths = pad.fit_to_box(
            draft, src_rect=src_rect, padding=planner.padding_mm, penup_mm=0
        )  # self.penup_mm)
        # print('Paths before', paths)
        paths = [
            np.hstack([P[:, :2], radius_to_z(self.params, P[:, 2]).reshape(-1, 1)])
            for P in paths
        ]
        # print('Paths after', paths)
        self.trajectories = pad.fit_to_frame(paths)
        self.src_rect = src_rect

        # print('Trajs', self.trajectories)
        self.trajectories_computed = False
        self.penup_flags = [False for _ in self.trajectories]
        self.chunk_indices = [i for i in range(len(self.trajectories))]


class TrajectoryPlanner:
    def __init__(self, itiba, **kwargs):
        self.itiba = itiba
        self.params = dict(
            planner="Trapezoidal",
            dce_thresh_mm=100,
            jerk_iterations=450,
            topp_tol_mm=100.0,
            isotropy=1.0,
            angle=50.0,
            order=3,
            smoothing=0.2,
            max_depth=4.0,
            bspline_degree=5,
            resample_topp=False,
            sorted=True,
            bezier_split_angle=54,
        )
        self.params.update(kwargs)
        try:
            self.params["tool_info"] = itiba.tool_info
        except AttributeError as e:
            raise AttributeError(
                "You need to select a tool in the interface to use the planner"
            )

        self.params = edict(self.params)
        self.padding_mm = 0.0
        self.penup_mm = 50.0
        self.plane_offset_mm = 0.0
        self.offset_mm = [0.0, 0.0]

        if "plane_offset_mm" in kwargs:
            self.plane_offset_mm = kwargs["plane_offset_mm"]
        self.itiba.update_handeye()

        self.trajectories = []
        self.trajectories_computed = False
        self.draft = []
        self.penup_flags = []
        self.src_rect = None

    def load_paths(
        self, path, max_paths=None, dest_rect=None, src_rect=None, select_indices=[]
    ):
        self.draft = load(path, self.params)
        if self.draft is None:
            return
        self.fit_draft(dest_rect, max_paths, select_indices, src_rect=src_rect)

    def fit_draft(
        self, dest_rect=None, max_paths=None, select_indices=[], src_rect=None
    ):
        pad = self.itiba.pad
        pad.offset = (
            self.plane_offset_mm / 1000
        )  # + self.itiba.tool_info.offset_mm / 1000
        pad.pos_offset = np.array(self.offset_mm) / 1000
        pad.update()
        if dest_rect is not None:
            pad.scaled_box = dest_rect
        paths = pad.fit_to_box(
            self.draft, src_rect=src_rect, padding=self.padding_mm, penup_mm=0
        )  # self.penup_mm)
        # print('Paths before', paths)
        paths = [
            np.hstack([P[:, :2], radius_to_z(self.params, P[:, 2]).reshape(-1, 1)])
            for P in paths
        ]
        # print('Paths after', paths)
        self.trajectories = pad.fit_to_frame(paths)
        self.src_rect = src_rect

        if select_indices:
            self.trajectories = [self.trajectories[i] for i in select_indices]
            self.draft = [self.draft[i] for i in select_indices]
        if max_paths is not None:
            if max_paths < 0:
                self.draft = self.draft[max_paths:]
                self.trajectories = self.trajectories[max_paths:]
            else:
                self.draft = self.draft[:max_paths]
                self.trajectories = self.trajectories[:max_paths]

        # print('Trajs', self.trajectories)
        self.trajectories_computed = False
        self.penup_flags = [False for _ in self.trajectories]
        self.chunk_indices = [i for i in range(len(self.trajectories))]

    def compute_trajectories(self, debug=False):
        params = self.params

        max_speed = cfg.cartesian_speed
        accel_time = cfg.acceleration_time
        trajectory_includes_penup = False

        # Select planner from parameters
        if params.planner == "Trapezoidal":
            traj_callback = plan_trapezoidal
        elif params.planner == "Minimum Jerk":
            traj_callback = plan_min_jerk
        elif params.planner == "Calligraphic":
            traj_callback = plan_calligraphic
            trajectory_includes_penup = True
            max_speed *= 2
        elif params.planner == "Test simplify":
            traj_callback = plan_simplify_test
        elif params.planner == "Thick bezier":
            traj_callback = plan_thick_resample_bezier
        elif params.planner == "Pass":
            traj_callback = plan_pass
        elif params.planner == "Thick":
            traj_callback = plan_thick_resample
        elif params.planner == "Thick BSpline":
            traj_callback = plan_thick_resample_bspline
        elif params.planner == "Calligraph":
            traj_callback = plan_thick_resample_bspline_calligraph
        else:
            raise ValueError("Unknown planner" + params.planner)
        if traj_callback is None:
            raise ValueError("No trajectory callback")

        # print("Planning draft with %d paths", len(self.draft))
        draft = self.itiba.pad.fit_to_box(
            self.draft, src_rect=self.src_rect, padding=self.padding_mm, penup_mm=0.0
        )
        log("Resulting fitted draft", len(draft))
        trajectories, self.penup_flags = plan_polylines_and_penups(
            traj_callback,
            draft,
            max_speed,
            accel_time,
            self.penup_mm / 1000,
            trajectory_includes_penup,
            params,
            debug=debug,
            get_penup_flags=True,
        )
        log("Resulting planned trajectories", len(trajectories))

        def adapt_xy(P):
            if type(P) == list:
                return [adapt_xy(Q) for Q in P]
            return np.vstack([-P[:, 1], -P[:, 0]]).T

        self.trajectories_2d = adapt_xy([np.array(P)[:, :2] for P in trajectories])
        self.draft_fitted = adapt_xy([np.array(P)[:, :2] for P in draft])

        self.trajectories = self.itiba.pad.fit_to_frame(trajectories)
        self.trajectories_computed = True
        self.chunk_indices = [0]

    def compute_and_plan_trajectories(self, debug=False):
        raise ValueError("Deprecated function")
        self.compute_trajectories(debug)

        cur_pos = self.itiba.robot.pen_tcp_pos(self.itiba.robot.q)
        trajectories = self.itiba.pad.augment_with_pen_orientation(
            self.trajectories, self.itiba.robot, self.itiba.robot.qr
        )
        trajectories += [
            tu.trapezoidal_approach(trajectories[-1][-1], cur_pos)
        ]  # [tu.approach(cur_pos, trajectories[-1][-1])]
        self.itiba.plan_task_trajectories(trajectories)
        self.task_trajectories_augmented = trajectories

    def plan_cartesian(self):
        params = self.params

        max_speed = cfg.cartesian_speed
        accel_time = cfg.acceleration_time
        trajectory_includes_penup = False

        # Select planner from parameters
        if params.planner == "Trapezoidal":
            traj_callback = plan_trapezoidal
        elif params.planner == "Minimum Jerk":
            traj_callback = plan_min_jerk
        elif params.planner == "Calligraphic":
            traj_callback = plan_calligraphic
            trajectory_includes_penup = True
            max_speed *= 2
        elif params.planner == "Test simplify":
            traj_callback = plan_simplify_test
        elif params.planner == "Thick bezier":
            traj_callback = plan_thick_resample_bezier
        elif params.planner == "Pass":
            traj_callback = plan_pass
        elif params.planner == "Thick":
            traj_callback = plan_thick_resample
        elif params.planner == "Thick BSpline":
            traj_callback = plan_thick_resample_bspline
        elif params.planner == "Calligraph":
            traj_callback = plan_thick_resample_bspline_calligraph

        if traj_callback is None:
            raise ValueError("No trajectory callback")

        log("Planning draft with %d paths", len(self.draft))
        draft = self.itiba.pad.fit_to_box(
            self.draft, src_rect=self.src_rect, padding=self.padding_mm, penup_mm=0.0
        )
        log("Resulting fitted draft", len(draft))
        trajectories = plan_polylines(
            traj_callback,
            draft,
            max_speed,
            accel_time,
            self.penup_mm / 1000,
            params=params,
        )
        return trajectories

    def compute_and_plan(self):
        self.compute_and_plan_with_dips(distance=np.inf, predip=False)

    def compute_and_plan_with_dips(
        self, distance=1.0, predip=False, dip_command="Dip-0", pen_accel_time=None
    ):
        params = self.params

        max_speed = cfg.cartesian_speed
        accel_time = cfg.acceleration_time
        if pen_accel_time is None:
            pen_accel_time = accel_time
        trajectory_includes_penup = False
        print("Max speed is ", max_speed)
        print("Accel time is", accel_time)

        # Select planner from parameters
        if params.planner == "Trapezoidal":
            traj_callback = plan_trapezoidal
        elif params.planner == "Minimum Jerk":
            traj_callback = plan_min_jerk
        elif params.planner == "Calligraphic":
            traj_callback = plan_calligraphic
            trajectory_includes_penup = True
            max_speed *= 2
        elif params.planner == "Test simplify":
            traj_callback = plan_simplify_test
        elif params.planner == "Thick bezier":
            traj_callback = plan_thick_resample_bezier
        elif params.planner == "Pass":
            traj_callback = plan_pass
        elif params.planner == "Thick":
            traj_callback = plan_thick_resample
        elif params.planner == "Thick BSpline":
            traj_callback = plan_thick_resample_bspline
        elif params.planner == "Calligraph":
            traj_callback = plan_thick_resample_bspline_calligraph
        else:
            raise ValueError("Unknown planner " + params.planner)

        if traj_callback is None:
            raise ValueError("No trajectory callback")

        log("Planning draft with %d paths", len(self.draft))
        draft, mat = self.itiba.pad.fit_to_box(
            self.draft,
            src_rect=self.src_rect,
            padding=self.padding_mm,
            penup_mm=0.0,
            get_mat=True,
        )
        s_factor = np.sqrt(
            abs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
        )  # np.linalg.det(mat[:2,:2]))
        print("Scale factor: ", s_factor)
        log("Resulting fitted draft", len(draft))
        trajectories = plan_polylines(
            traj_callback,
            draft,
            max_speed,
            accel_time,
            self.penup_mm / 1000,
            params=params,
        )

        # pdb.set_trace()
        # Split into dip chunks
        d = 0.0
        chunks = [[]]
        for traj in trajectories:
            log("Chunk with size", traj.shape)
            chunks[-1].append(traj)
            d += geom.chord_length(traj)
            if d > distance:
                log("Total distance: ", d)
                chunks.append([])
                d = 0
        log("Total distance: ", d)

        chunks = [paths for paths in chunks if paths]
        # chunks = [(paths, [False for _ in paths]) for paths in chunks]
        # pdb.set_trace()
        # Add penups within chunks
        chunks = [
            insert_penups(paths, self.penup_mm / 1000, pen_accel_time)
            for paths in chunks
            if paths
        ]
        # Add approaches
        chunks = [
            append_penups(paths, penups, self.penup_mm / 1000, pen_accel_time)
            for paths, penups in chunks
        ]
        # Add indices for visualization
        chunk_indices = [[i for _ in chunk[0]] for i, chunk in enumerate(chunks)]

        self.chunk_indices = sum(chunk_indices, [])

        # print(len(self.chunk_indices), 'Chunk indices', [c for c in chunk_indices])

        chunk_penups = [chunk[1] for chunk in chunks]
        chunk_paths = [chunk[0] for chunk in chunks]

        # Flattened
        # log("Resulting planned trajectories", len(trajectories))
        # def adapt_xy(P):
        #     if type(P)==list:
        #         return [adapt_xy(Q) for Q in P]
        #     return np.vstack([-P[:,1], -P[:,0]]).T

        # self.trajectories_2d = adapt_xy([np.array(P)[:,:2] for P in trajectories])
        # self.draft_fitted = adapt_xy([np.array(P)[:,:2] for P in draft])

        # for c in chunks:
        #    log("Shapes", [P.shape for P in c])
        chunk_paths = [self.itiba.pad.fit_to_frame(paths) for paths in chunk_paths]
        # pdb.set_trace()
        self.trajectories = sum(chunk_paths, [])
        self.penup_flags = sum(chunk_penups, [])
        chunk_paths = [
            self.itiba.pad.augment_with_pen_orientation(
                paths, self.itiba.robot, self.itiba.robot.qr
            )
            for paths in chunk_paths
        ]

        self.itiba.clear_motions()
        robot = self.itiba.robot
        initial_q = robot.q

        print("Computing chunk trajectories")
        for i, chunk in enumerate(chunk_paths):
            if self.itiba.motions:
                q = self.itiba.Qs[-1]
            else:
                q = robot.q
            if i != 0 or predip:
                speed_factor = cfg.speed_factor
                cfg.speed_factor = 1.0
                self.itiba.plan_script(dip_command, append_to_configuration=q)
                cfg.speed_factor = speed_factor
                if not self.itiba.trajectory_status:
                    log("Error computing dip trajectory")
                    break
                else:
                    log("Dip OK")
                if self.itiba.motions:
                    q = self.itiba.Qs[-1]
                else:
                    q = robot.q
            Qs, dQs = self.itiba.compute_task_trajectories(chunk, q=q, stiff_orn=False)
            self.itiba.trajectory_status = robot.check_joint_trajectory(Qs, dQs)
            if not self.itiba.trajectory_status:
                log("Error computing chunk trajectory")
                # break
            self.itiba.motions.append(Motion(Qs, dQs))

        if self.itiba.trajectory_status:
            if self.itiba.motions:
                q = self.itiba.Qs[-1]
            else:
                q = robot.q
            # Add final approach to original position
            Qs, dQs = tu.joint_approach(q, initial_q)
            self.itiba.trajectory_status = robot.check_joint_trajectory(Qs, dQs)
            if self.itiba.trajectory_status:
                self.itiba.motions.append(Motion(Qs, dQs))
                self.trajectories_computed = True
            else:
                log("Error computing return to initial trajectory")

        print("Supposed to plot")
        if not self.itiba.trajectory_status or self.itiba.safe_mode:
            self.itiba.plot_trajectory_stats()

    def plot_plan_image(
        self,
        backend="plt",
        show=False,
        figsize=(7, 7),
        color=None,
        alpha=0.8,
        image_size=None,
    ):
        self.itiba.update_handeye()
        handeye = self.itiba.handeye

        mat = np.eye(3)
        mat = mat @ geom.scaling_2d([1, -1])
        mat = mat @ geom.scaling_2d([-1, 1])
        mat = geom.rot_2d(geom.radians(-90)) @ mat
        flipm = mat

        def flip_transform(P):
            if P.shape[1] > 2:
                m = np.eye(4)
                m[:2, :2] = flipm[:2, :2]
                return geom.affine_transform(m, P)
            return geom.affine_transform(flipm, P)

        if show:
            ply.figure(figsize=figsize)
            ply.subplot(1, 1)
        if handeye is not None and handeye.cam_image is not None:
            cam_image = np.array(handeye.cam_image)
            if self.itiba.handeye_valid():
                size = 50 / 1000
                points = [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]]
                projected_points = handeye.project_points(points, "aruco")

                ply.image(cam_image)
                ply.line(projected_points[0], projected_points[1], "r")
                ply.line(projected_points[0], projected_points[2], "g")
                ply.line(projected_points[0], projected_points[3], "b")
        if True:
            pad = self.itiba.pad
            if pad.is_valid():
                points = pad.get_box_points()
                if False:  # self.itiba.handeye_valid():
                    projected_points = handeye.project_points(points, "world")
                else:
                    m = np.linalg.inv(pad.get_offset_frame())  # frame)
                    projected_points = geom.affine_transform(m, points)
                    projected_points = flip_transform(projected_points)

                ply.rectangle(
                    flip_transform(np.array(pad.box)), color="c", label="Work rect"
                )
                ply.rectangle(
                    flip_transform(np.array(pad.scaled_box)),
                    color="r",
                    label="Drawing rect",
                )
                # ply.stroke(projected_points[:,:2], 'r', label='Work rect')

                for i, (trajectory, penup) in enumerate(
                    zip(self.trajectories, self.penup_flags)
                ):
                    if penup:
                        continue
                    if False:  # self.itiba.handeye_valid():
                        projected_points = handeye.project_points(trajectory, "world")
                    else:
                        m = np.linalg.inv(pad.get_offset_frame())  # frame)
                        projected_points = geom.affine_transform(m, trajectory)
                        projected_points = flip_transform(projected_points)
                        # print('flip', projected_points)

                    if color is None:
                        clr = ply.categorical_color(self.chunk_indices[i])
                    else:
                        clr = color
                    if trajectory.shape[1] > 2:
                        # print('z', projected_points[:,2])
                        w = z_to_radius(self.params, projected_points[:, 2])
                        # print('z', projected_points[:,2])
                        # print('radius', w)
                        points = projected_points[:, :2]
                        ply.fill(
                            geom.thick_curve(np.hstack([points, w.reshape(-1, 1)])),
                            clr,
                            alpha=alpha,
                        )
                    else:
                        ply.stroke(
                            projected_points[:, :2],
                            clr,
                            label="Trajectory" if i == 0 else "",
                            lw=2,
                            alpha=alpha,
                        )
                    if not self.trajectories_computed:
                        ply.text("%d" % i, projected_points[0][:2], color="k", size=7)
        if show:
            ply.setup_2d(flip_y=False)
            return ply.show(backend)

    def get_plan_image(self):
        from py5canvas import canvas

        self.itiba.update_handeye()
        handeye = self.itiba.handeye

        if handeye is not None and handeye.cam_image is not None:
            cam_image = np.array(handeye.cam_image)
            w = cam_image.shape[1]
            h = cam_image.shape[0]
            if self.itiba.handeye_valid():
                size = 50 / 1000
                points = [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]]
                projected_points = handeye.project_points(points, "aruco")

                cc = canvas.Canvas(w, h)
                cc.image(cam_image)
                cc.no_fill()
                cc.stroke_weight(1)
                cc.stroke(255, 0, 0)
                # c.circle(200, 200, 100)
                cc.line(projected_points[0], projected_points[1])
                cc.stroke(0, 255, 0)
                cc.line(projected_points[0], projected_points[2])
                cc.stroke(0, 0, 255)
                cc.line(projected_points[0], projected_points[3])
            if True:
                pad = self.itiba.pad
                if pad.is_valid():
                    points = pad.get_box_points()
                    if self.itiba.handeye_valid():
                        projected_points = handeye.project_points(points, "world")
                    else:
                        mat = np.linalg.inv(pad.get_offset_frame())  # pad.frame)
                        projected_points = mat @ points

                    cc.polygon(projected_points)

                    for i, trajectory in enumerate(self.trajectories):
                        if self.itiba.handeye_valid():
                            projected_points = handeye.project_points(
                                trajectory, "world"
                            )
                        else:
                            mat = np.linalg.inv(pad.get_offset_frame())  # frame)
                            projected_points = mat @ trajectory

                        cc.stroke(255, 0, 0, 128)
                        cc.stroke_weight(3.0)
                        cc.polyline(projected_points)

                cam_image = cc.get_image()
                return Image.fromarray(cam_image)
            else:
                log("Invalid handeye")
            ref_h = 180
            # c.image(cam_image, 0, c.height-ref_h, (w/h)*ref_h, ref_h)

        log("Handeye not configured")
        return None


def plan_trapezoidal(samples, max_speed, accel_time, penup_distance, params):
    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    paths, velocities_, accelerations_ = tu.plan_polylines_time_optimal(
        samples,
        max_speed=max_speed,
        accel_time=accel_time,
        tol=params.topp_tol_mm / 1000,
    )
    return paths, samples


def ema(data, alpha):
    ema_values = [data[0]]  # Initialize EMA with the first data point

    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])

    return np.array(ema_values)


def plan_calligraphic(samples, max_speed, accel_time, penup_distance, params):
    raise ValueError("Plan calligraphic needs fixing (penups/z) or to be deprectated")
    simplified = [tu.simplify.dce(P, params.dce_thresh_mm / 1000) for P in samples]
    simplified = [P for P in simplified if len(P) > 1]
    if not simplified:
        return [], []

    paths = [np.hstack([P, np.zeros((len(P), 1))]) for P in simplified]
    up = np.array([0, 0, penup_distance])
    # Square pen ups
    # motor_plan = [paths[0][0] + up, paths[0]]
    # if len(paths) > 1:
    #     for i in range(1, len(paths)):
    #         P1 = paths[i-1]
    #         P2 = paths[i]
    #         if geom.distance(P1[-1], P2[0]) > 2e-3:
    #             motor_plan.append(P1[-1] + up)
    #             motor_plan.append(P2[0] + up)
    #         motor_plan.append(P2)
    # motor_plan.append(paths[-1][-1] + up)
    # Triangle pen ups
    motor_plan = [paths[0]]
    if len(paths) > 1:
        for i in range(1, len(paths)):
            P1 = paths[i - 1]
            P2 = paths[i]
            if geom.distance(P1[-1], P2[0]) > 2e-3:
                motor_plan.append((P1[-1] + P2[0]) / 2 + up)
            motor_plan.append(P2)
    # motor_plan.append(paths[-1][-1] + up)

    motor_plan = np.vstack(motor_plan)
    # Add stop points
    motor_plan = np.vstack([motor_plan[0], motor_plan, motor_plan[-1]])
    trajectories = tu.plan_polylines_lqt(
        [motor_plan],
        max_speed,
        accel_time=accel_time,
        isotropy=params.isotropy,
        sigma=params.smoothing,
        penups=True,
        order=params.order,
        opt_x0=False,
        subd=5,
        angle=params.angle,
    )
    # Clip zs to zero
    log("Deviation from base: ", np.min(trajectories[0][:, 2]))
    # trajectories[0][:,2] = ema(np.maximum(trajectories[0][:,2], 0.0), 0.1)
    return trajectories, simplified


def plan_min_jerk(samples, max_speed, accel_time, penup_distance, params):
    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    # chains = [np.array(chain[:,:2]) for chain in samples]
    # widths = [np.array(chain[:,2]) for chain in samples]
    # samples = [np.hstack([P, radius_to_z(params, w).reshape(-1,1)]) for P, w in zip(chains, widths)]
    paths, simplified = tu.plan_polylines_jerk(
        samples,
        params.dce_thresh_mm / 1000,
        max_speed=max_speed,
        max_accel_time=accel_time,
        maxiter=params.jerk_iterations,
    )
    return paths, simplified


def plan_simplify_test(samples, max_speed, accel_time, penup_distance, params):
    simplified = [tu.simplify.dce(P, params.dce_thresh_mm / 1000) for P in samples]
    return samples, simplified


def plan_thick_resample(samples, max_speed, accel_time, penup_distance, params):
    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    trajectories, _, _ = tu.plan_polylines_time_optimal(
        samples, cfg.cartesian_speed, cfg.acceleration_time
    )
    return trajectories, [P[:, :2] for P in samples]


def remap(x, input_range, output_range):
    return np.interp(
        x, input_range, output_range, left=output_range[0], right=output_range[1]
    )


def radius_to_z(params, r):
    wrange = [
        params.tool_info.min_radius_mm / 1000,
        params.tool_info.max_radius_mm / 1000,
    ]
    # zrange = [0.0+params.tool_info.offset_mm/1000, -params.tool_info.range_mm/1000+params.tool_info.offset_mm/1000]
    zrange = [0.0, -params.tool_info.range_mm / 1000]
    return remap(r, wrange, zrange) + (params.tool_info.offset_mm / 1000)


def z_to_radius(params, z):
    wrange = [
        params.tool_info.min_radius_mm / 1000,
        params.tool_info.max_radius_mm / 1000,
    ]
    # zrange = [0.0+params.tool_info.offset_mm/1000, -params.tool_info.range_mm/1000+params.tool_info.offset_mm/1000]
    zrange = [0.0, -params.tool_info.range_mm / 1000]
    return remap(z - (params.tool_info.offset_mm / 1000), zrange[::-1], wrange[::-1])


def plan_thick_resample_bspline(samples, max_speed, accel_time, penup_distance, params):
    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    Cps_3d = samples

    if params.resample_topp:
        trajectories, _, _ = tu.plan_polylines_time_optimal(
            [tu.bspline_samples(P, len(P) * 7, params.bspline_degree) for P in Cps_3d],
            cfg.cartesian_speed,
            cfg.acceleration_time,
        )
    else:
        trajectories = [
            tu.resample_bspline(P, params.bspline_degree, max_speed, accel_time)
            for P in Cps_3d
        ]
    return trajectories, [P[:, :2] for P in samples]


def plan_thick_resample_bspline_calligraph(
    samples, max_speed, accel_time, penup_distance, params
):
    from calligraph import bspline

    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    Cps_3d = samples

    if params.resample_topp:
        trajectories, _, _ = tu.plan_polylines_time_optimal(
            [
                bspline.bspline_samples(P, len(P) * 7, params.bspline_degree + 1)
                for P in Cps_3d
            ],
            cfg.cartesian_speed,
            cfg.acceleration_time,
        )
    else:
        trajectories = [
            bspline.resample_bspline(
                P, cfg.dt, params.bspline_degree, max_speed * 0.5, None
            )
            for P in Cps_3d
        ]
    return trajectories, [P[:, :2] for P in samples]


def plan_thick_resample_bezier(samples, max_speed, accel_time, penup_distance, params):
    samples = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    chains_3d = samples
    print("Resampling bezier")
    if False:  # params.resample_topp:
        trajectories, _, _ = tu.plan_polylines_time_optimal(
            chains_3d, cfg.cartesian_speed, cfg.acceleration_time
        )
    else:
        trajectories = [
            tu.resample_bezier(chain, max_speed, accel_time) for chain in chains_3d
        ]
    return trajectories, [P[:, :2] for P in samples]


def plan_pass(samples, max_speed, accel_time, penup_distance, params):
    trajectories = [
        np.hstack([P[:, :2], radius_to_z(params, P[:, 2]).reshape(-1, 1)])
        for P in samples
    ]
    return trajectories, samples


def append_penups(paths, penup_flags, penup_distance_m, accel_time):
    start = paths[0][0] + np.array([0, 0, penup_distance_m])
    end = paths[-1][-1] + np.array([0, 0, penup_distance_m])
    arrival = np.array(tu.rest_inout(tu.approach(start, paths[0][0]), 0.25))
    departure = np.array(tu.rest_inout(tu.approach(paths[-1][-1], end), 0.25))
    return [arrival[:, :3]] + paths + [departure[:, :3]], [True] + penup_flags + [True]


def insert_penups(paths, penup_distance, accel_time):
    trajectories = [paths[0]]
    penup_flags = [False]
    if len(paths) > 1:
        for i in range(len(paths) - 1):
            P1 = paths[i]
            P2 = paths[i + 1]
            dist = geom.distance(P1[-1], P2[0])
            if dist > 1e-3:
                # pdb.set_trace()
                trajectories.append(
                    tu.lifting_motion(
                        P1[-1], P2[0], penup_distance, accel_time=accel_time
                    )
                )
                penup_flags.append(True)
                # penup_distance.append(True)
                # distances_down.append(0.0)
            # else:

            trajectories.append(P2)
            penup_flags.append(False)

    return trajectories, penup_flags


def plan_polylines(
    planning_callback,
    samples,
    max_speed,
    accel_time,
    penup_distance,
    trajectory_includes_penups=False,
    params={},
):
    if samples[0].shape[1] < 3:
        samples = [np.hstack([P, np.zeros((len(P), 1))]) for P in samples]
    # up = np.array([0, 0, penup_distance])
    # samples = ([np.array([samples[0][0]+up, samples[0][0]])]
    #                     + samples
    #                     + [np.array([samples[-1][-1], samples[-1][-1]+up])])
    paths, simplified = planning_callback(
        samples,
        penup_distance=penup_distance,
        max_speed=max_speed,
        accel_time=accel_time,
        params=params,
    )
    # Remove corrupt paths
    paths = [P for P in paths if len(P) > 1]

    print("####################################")
    print("Planned trajectories with durations:")
    max_speed = 0
    for P in paths:
        print(len(P) * cfg.dt, "seconds")
        D = np.diff(P, axis=0) / cfg.dt
        try:
            max_speed = max(max_speed, np.max(np.linalg.norm(D, axis=1)))
        except ValueError as e:
            print(e)
            breakpoint()
    print("Max speed:", max_speed)
    print("####################################")

    return paths


def plan_polylines_and_penups(
    planning_callback,
    samples,
    max_speed,
    accel_time,
    penup_distance,
    trajectory_includes_penups=False,
    get_penup_flags=False,
    params={},
    debug=True,
):
    """General function to plan a 2d trajectory so it is projected to 3d on the drawing pad
    Does not handle orientation, only position
    """
    # config.progress_text = 'Planning trajectory'
    # config.progress = 0
    # config.progress_max = len(samples)
    distances_down = []
    if not trajectory_includes_penups:
        # Plan each trajectory and then concatenate lifting motions
        # If trajectory is 2d augment
        if samples[0].shape[1] < 3:
            samples = [np.hstack([P, np.zeros((len(P), 1))]) for P in samples]
        # up = np.array([0, 0, penup_distance])
        # samples = ([np.array([samples[0][0]+up, samples[0][0]])]
        #                     + samples
        #                     + [np.array([samples[-1][-1], samples[-1][-1]+up])])
        paths, simplified = planning_callback(
            samples,
            penup_distance=penup_distance,
            max_speed=max_speed,
            accel_time=accel_time,
            params=params,
        )
        print("####################################")
        print("Planned trajectories with durations:")
        max_speed = 0
        for P in paths:
            print(len(P) * cfg.dt, "seconds")
            D = np.diff(P, axis=0) / cfg.dt
            max_speed = max(max_speed, np.max(np.linalg.norm(D, axis=1)))
        print("Max speed:", max_speed)
        print("####################################")

        paths = [P for P in paths if len(P) > 1]

        log("lens", [len(P) for P in paths])
        if not paths:
            if get_penup_flags:
                return [], []
            return []
        trajectories = [paths[0]]
        penup_flags = [False]
        if len(paths) > 1:
            for i in range(len(paths) - 1):
                P1 = paths[i]
                P2 = paths[i + 1]
                if geom.distance(P1[-1], P2[0]) > 2e-3:
                    trajectories.append(
                        tu.lifting_motion(
                            P1[-1], P2[0], penup_distance, accel_time=accel_time
                        )
                    )
                    penup_flags.append(True)
                    # distances_down.append(0.0)
                trajectories.append(P2)
                penup_flags.append(False)
                # distances_down.append(geom.chord_length(P2))

    else:
        # Create one motor plan with pen-ups and plan single trajectory
        trajectories, simplified = planning_callback(
            samples,
            penup_distance=penup_distance,
            max_speed=max_speed,
            accel_time=accel_time,
            params=params,
        )
        print("####################################")
        print("Planned trajectories with durations:")
        max_speed = 0
        for P in trajectories:
            print(len(P) * cfg.dt, "seconds")
            D = np.diff(P, axis=0) / cfg.dt
            max_speed = max(max_speed, np.max(np.linalg.norm(D, axis=1)))
        print("Max speed:", max_speed)
        print("####################################")
        # distances_down = [geom.chord_length(traj) for traj in trajectories]
        penup_flags = [False for _ in trajectories]
    velocities = [np.diff(P, axis=0) / cfg.dt for P in trajectories]
    accelerations = [np.diff(P, n=2, axis=0) / (cfg.dt**2) for P in trajectories]
    max_accel = max_speed / accel_time

    if debug:

        def adapt_xy(P):
            if type(P) == list:
                return [adapt_xy(Q) for Q in P]
            return np.vstack([-P[:, 1], P[:, 0]]).T

        ply.figure(title="Cartesian trajectory")

        Tv = 0.0
        Ta = 0.0
        lw = 3
        for i, X in enumerate(trajectories):
            # speed = velocities[i] #tu.compute_speed(X, cfg.dt)
            # accel = accelerations[i] #tu.compute_accel(X, cfg.dt)
            tv = (
                np.linspace(0, (len(velocities[i]) - 1) * cfg.dt, len(velocities[i]))
                + Tv
            )
            ta = (
                np.linspace(
                    0, (len(accelerations[i]) - 1) * cfg.dt, len(accelerations[i])
                )
                + Ta
            )
            ply.subplot(1, 1, title="Velocity")
            ply.plot(tv, velocities[i][:, 0], color=ply.categorical_color(i), lw=lw)
            ply.plot(tv, velocities[i][:, 1], color=ply.categorical_color(i), lw=lw)
            ply.plot(tv, velocities[i][:, 2], color="b", alpha=0.5)
            ply.subplot(1, 2, title="Acceleration")
            ply.plot(ta, accelerations[i][:, 0], color=ply.categorical_color(i), lw=lw)
            ply.plot(ta, accelerations[i][:, 1], color=ply.categorical_color(i), lw=lw)
            ply.plot(ta, accelerations[i][:, 2], color="b", alpha=0.5)
            Tv = tv[-1] + cfg.dt
            Ta = ta[-1] + cfg.dt

        ply.subplot(1, 1, title="Velocity")
        ply.rectangle([0, -max_speed], [Tv, max_speed], color="k", alpha=0.1, fill=True)
        ply.subplot(1, 2, title="Acceleration")
        ply.rectangle([0, -max_accel], [Ta, max_accel], color="k", alpha=0.1, fill=True)

        ply.subplot(2, 1, title="Trajectory 2d")
        ply.stroke(adapt_xy(samples), "k", alpha=0.25, lw=lw)
        for i in range(len(trajectories)):
            ply.stroke(adapt_xy(trajectories[i]), ply.categorical_color(i), lw=lw)
        ply.axis_equal(True)

        ply.subplot(2, 2, title="Trajectory 3d")

        box = geom.bounding_box(trajectories)

        for X in trajectories:
            X = np.array(X)
            vals = (X[:, 2]) / (box[1][2] - box[0][2])
            ply.stroke3d(
                X,
                color=[
                    ply.cmap(z, "magma") if z >= 0 else ply.cmap(-z, "viridis")
                    for z in vals
                ],
            )
        ply.show()
    if get_penup_flags:
        return trajectories, penup_flags
    return trajectories


def load(path, params):
    if ".svg" in path:
        return load_svg(path, params)
    else:
        return load_thick_paths(path, bspline="bspline" in params.planner.lower())


def load_thick_paths(path, bspline=False):
    strokes = utils.load_json(path)
    if not strokes:
        return
    if bspline:
        chains = [np.array(P) for P in strokes["control_points"]]
        widths = [np.array(w) for w in strokes["control_widths"]]
    else:
        chains = [np.array(P) for P in strokes["points"]]
        widths = [np.array(w) for w in strokes["stroke_widths"]]
    if not len(widths[0].shape):
        widths = [np.zeros(len(P)) for P in chains]
    return [np.hstack([P, w.reshape(-1, 1)]) for P, w in zip(chains, widths)]


def load_svg(path, params):
    log("Loading SVG:", path)
    try:
        draft = svg.load_svg_bezier_chains(path)
    except FileNotFoundError as e:
        log(e)
        return
    zz = svg.load_svg_polylines(path)

    dest_rect = geom.make_rect(0, 0, 1024, 768)
    draft_in = geom.transform_to_rect(draft, dest_rect)
    if params.sorted:
        draft_in = spatial.sort_paths(draft_in)
    draft = []
    tol = 0.2
    for chain in draft_in:  # [2:4]:
        if params.bezier_split_angle > 0:
            chunks = bezier.split_chain(chain, angle_tol=params.bezier_split_angle)
        else:
            chunks = [chain]
        chunk_beziers = []
        for chunk in chunks:
            chunk_beziers.append(chunk)
            # pts = bezier.cubic_bezier_adaptive(chunk, tol)
            # pts = bezier.bezier_piecewise(chunk, 30)
            # beziers = fitCurve(pts, 4)
            # chunk_beziers += beziers #.append(beziers)

        for chain in chunk_beziers:
            # chain = bezier.beziers_to_chain(chunk_beziers)
            P = bezier.cubic_bezier_adaptive(chain, tol)
            # P = geom.adaptive_subdivide(P, 5)
            draft += [P]  # geom.split_polyline_at_angle(P, 160)
        # draft.append(P)
        # log('chain len:', len(chain))
        # draft.append(bezier.beziers_to_chain(beziers))

    return draft
