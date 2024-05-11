from importlib import reload
from . import config, geom, utils, bezier, svg, spatial
from . import trajectory_utils as tu
from . import matplotlyb as ply
from easydict import EasyDict as edict
from PIL import Image
import numpy as np
reload(tu)
cfg = config.cfg

class TrajectoryPlanner:
    def __init__(self, itiba, **kwargs):
        self.itiba = itiba
        self.params = dict(planner='Trapezoidal',
                           dce_thresh_mm=100,
                            jerk_iterations=450,
                            topp_tol_mm=100.0,
                            isotropy=1.0,
                            angle=50.0,
                            order=3,
                            smoothing=0.2,
                            max_depth=4.0,
                            resample_topp=False,
                            sorted=True,
                            bezier_split_angle=54)
        self.params.update(kwargs)
        self.params = edict(self.params)
        self.padding_mm = 0.0
        self.penup_mm = 50.0
        self.plane_offset_mm = 0.0
        self.offset_mm = [0.0, 0.0]

        self.itiba.update_handeye()

        self.trajectories = []
        self.trajectories_computed = False
        self.draft = []
        
    def load_paths(self, path):
        self.draft = load(path, self.params)
        if self.draft is None:
            return
        self.fit_draft()

    def fit_draft(self):
        pad = self.itiba.pad
        pad.offset = self.plane_offset_mm / 1000 + self.itiba.pen_offset_base_mm / 1000
        pad.pos_offset = np.array(self.offset_mm)/1000
        pad.update()

        paths = pad.fit_to_box(self.draft, padding=self.padding_mm, penup_mm=self.penup_mm)
        self.trajectories = pad.fit_to_frame(paths)
        self.trajectories_computed = False

    def compute_trajectories(self, debug=False):
        params = self.params

        max_speed = cfg.cartesian_speed
        accel_time = cfg.acceleration_time
        trajectory_includes_penup = False

        # Select planner from parameters
        if params.planner == 'Trapezoidal':
            traj_callback = plan_trapezoidal
        elif params.planner == 'Minimum Jerk':
            traj_callback = plan_min_jerk
        elif params.planner == 'Calligraphic':
            traj_callback = plan_calligraphic
            trajectory_includes_penup = True
            max_speed *= 2
        elif params.planner == 'Test simplify':
            traj_callback = plan_simplify_test
        elif params.planner == 'Thick bezier':
            traj_callback = plan_thick_resample_bezier
        elif params.planner == 'Thick':
            traj_callback = plan_thick_resample
            
        if traj_callback is None:
            raise ValueError('No trajectory callback')

        print("Planning draft with %d paths", len(self.draft))
        draft = self.itiba.pad.fit_to_box(self.draft, padding=self.padding_mm, penup_mm=0.0)
        print("Resulting fitted draft", len(draft))
        trajectories = plan_polylines(traj_callback, draft, max_speed, accel_time,
                                        self.penup_mm/1000,
                                        trajectory_includes_penup,
                                        params,
                                        debug=debug)
        print("Resulting planned trajectories", len(trajectories))
        def adapt_xy(P):
            if type(P)==list:
                return [adapt_xy(Q) for Q in P]
            return np.vstack([-P[:,1], -P[:,0]]).T

        self.trajectories_2d = adapt_xy([np.array(P)[:,:2] for P in trajectories])
        self.draft_fitted = adapt_xy([np.array(P)[:,:2] for P in draft])

        self.trajectories = self.itiba.pad.fit_to_frame(trajectories)
        self.trajectories_computed = True

    def compute_and_plan_trajectories(self, debug=False):
        self.compute_trajectories(debug)

        cur_pos = self.itiba.robot.pen_tcp_pos(self.itiba.robot.q)
        trajectories = self.itiba.pad.augment_with_pen_orientation(self.trajectories, self.itiba.robot, self.itiba.robot.qr)
        trajectories += [tu.trapezoidal_approach(trajectories[-1][-1], cur_pos)] # [tu.approach(cur_pos, trajectories[-1][-1])]
        self.itiba.plan_task_trajectories(trajectories)
        self.task_trajectories_augmented = trajectories
 
    def compute_and_pan_with_dips(self, distance=1000.0):
        self.compute_trajectories(debug)


    def get_plan_image(self):
        from py5canvas import canvas
        self.itiba.update_handeye()
        handeye = self.itiba.handeye
        if handeye is not None and handeye.cam_image is not None:
            cam_image = np.array(handeye.cam_image)
            w = cam_image.shape[1]
            h = cam_image.shape[0]
            if self.itiba.handeye_valid():
                size = 50/1000
                points = [[0,0,0],
                        [size,0,0],
                        [0,size,0],
                        [0,0,size]]
                projected_points = handeye.project_points(points, 'aruco')
                cc = canvas.Canvas(w, h)
                cc.image(cam_image)
                cc.no_fill()
                cc.stroke_weight(1)
                cc.stroke(255, 0, 0)
                #c.circle(200, 200, 100)
                cc.line(projected_points[0], projected_points[1])
                cc.stroke(0, 255, 0)
                cc.line(projected_points[0], projected_points[2])
                cc.stroke(0, 0, 255)
                cc.line(projected_points[0], projected_points[3])

                pad = self.itiba.pad
                if pad.is_valid():
                    points = pad.get_box_points()
                    projected_points = handeye.project_points(points, 'world')
                    cc.polygon(projected_points)

                    for i, trajectory in enumerate(self.trajectories):
                        projected_points = handeye.project_points(trajectory, 'world')
                        cc.stroke(255, 0, 0, 128)
                        cc.stroke_weight(3.0)
                        cc.polyline(projected_points)
                        
                cam_image = cc.get_image()
                return Image.fromarray(cam_image)
            else:
                print("Invalid handeye")
            ref_h = 180
            #c.image(cam_image, 0, c.height-ref_h, (w/h)*ref_h, ref_h)
            
    
        print("Handeye not configured")
        return None
    

def plan_trapezoidal(samples,
                     max_speed,
                     accel_time,
                     penup_distance,
                     params):
    paths, velocities_, accelerations_ = tu.plan_polylines_time_optimal(samples,
                                                                  max_speed=max_speed,
                                                                  accel_time=accel_time,
                                                                        tol=params.topp_tol_mm/1000)
    return paths, samples

def ema(data, alpha):
    ema_values = [data[0]]  # Initialize EMA with the first data point

    for i in range(1, len(data)):
        ema_values.append(alpha * data[i] + (1 - alpha) * ema_values[-1])

    return np.array(ema_values)

def plan_calligraphic(samples,
                      max_speed,
                      accel_time,
                      penup_distance,
                      params):
    simplified = [tu.simplify.dce(P, params.dce_thesh_mm/1000) for P in samples]
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
            P1 = paths[i-1]
            P2 = paths[i]
            if geom.distance(P1[-1], P2[0]) > 2e-3:
                motor_plan.append((P1[-1] + P2[0])/2 + up)
            motor_plan.append(P2)
    #motor_plan.append(paths[-1][-1] + up)

    motor_plan = np.vstack(motor_plan)
    # Add stop points
    motor_plan = np.vstack([motor_plan[0], motor_plan,motor_plan[-1]])
    trajectories = tu.plan_polylines_lqt([motor_plan],
                                         max_speed,
                                         accel_time=accel_time,
                                        isotropy=params.isotropy,
                                        sigma=params.smoothing,
                                        penups=True,
                                        order=params.order,
                                        opt_x0=False,
                                        subd=5,
                                        angle=params.angle)
    # Clip zs to zero
    print('Deviation from base: ', np.min(trajectories[0][:,2]))
    #trajectories[0][:,2] = ema(np.maximum(trajectories[0][:,2], 0.0), 0.1)
    return trajectories, simplified

def plan_min_jerk(samples,
                 max_speed,
                 accel_time,
                 penup_distance,
                 params):
    paths, simplified = tu.plan_polylines_jerk(samples,
                                                                params.dce_thesh_mm/1000,
                                                                max_speed=max_speed,
                                                                max_accel_time=accel_time,
                                                                maxiter=params.max_jerk_iterations)
    return paths, simplified


def plan_simplify_test(samples,
                 max_speed,
                 accel_time,
                 penup_distance,
                 params):
    simplified = [tu.simplify.dce(P, params.dce_thesh_mm/1000) for P in samples]
    return samples, simplified

def plan_thick_resample(samples,
                        max_speed,
                        accel_time,
                        penup_distance,
                        params):
    chains = [np.array(chain[:,:2]) for chain in samples]
    widths = [np.array(chain[:,2]) for chain in samples]
    width_range_mm = [np.min(np.concatenate(widths))*1000, np.max(np.concatenate(widths))*1000]

    def map(x, input_range, output_range):
        return (output_range[0] + ((x - input_range[0])/(input_range[1] - input_range[0]))
                                    * (output_range[1] - output_range[0]))

    wrange = np.array(width_range_mm)/1000.0
    zrange = np.array([0.0, -params.max_depth])/1000.0

    def width_to_z(w):
        return map(w, wrange, zrange)

    def z_to_width(z):
        return map(z, zrange, wrange)

    chains_3d = [np.hstack([Cp, width_to_z(w.reshape(-1,1))]) for Cp, w in zip(chains, widths)]

    trajectories,_,_ = tu.plan_polylines_time_optimal(chains_3d, cfg.cartesian_speed, cfg.acceleration_time)
    return trajectories, [P[:,:2] for P in samples]

def plan_thick_resample_bezier(samples,
                        max_speed,
                        accel_time,
                        penup_distance,
                        params):
    chains = [np.array(chain[:,:2]) for chain in samples]
    widths = [np.array(chain[:,2]) for chain in samples]
    width_range_mm = [np.min(np.concatenate(widths))*1000, np.max(np.concatenate(widths))*1000]

    def map(x, input_range, output_range):
        return (output_range[0] + ((x - input_range[0])/(input_range[1] - input_range[0]))
                                    * (output_range[1] - output_range[0]))

    wrange = np.array(width_range_mm)/1000.0
    zrange = np.array([0.0, -params.max_depth])/1000.0

    def width_to_z(w):
        return map(w, wrange, zrange)

    def z_to_width(z):
        return map(z, zrange, wrange)

    chains_3d = [np.hstack([Cp, width_to_z(w.reshape(-1,1))]) for Cp, w in zip(chains, widths)]

    if params.resample_topp:
        trajectories,_,_ = tu.plan_bezier_chains_time_optimal(chains_3d, cfg.cartesian_speed, cfg.acceleration_time)
    else:
        trajectories = [tu.resample_bezier(chain, max_speed, accel_time) for chain in chains_3d]
    return trajectories, [P[:,:2] for P in samples]
    
def plan_polylines(planning_callback,
                   samples,
                   max_speed,
                   accel_time,
                   penup_distance,
                   trajectory_includes_penups=False,
                   params={},
                   debug=True):
    ''' General function to plan a 2d trajectory so it is projected to 3d on the drawing pad
    Does not handle orientation, only position
    '''
    #config.progress_text = 'Planning trajectory'
    #config.progress = 0
    #config.progress_max = len(samples)
    distances_down = []
    if not trajectory_includes_penups:
        # Plan each trajectory in 2d and then concatenate lifting motions
        if samples[0].shape[1] < 3:
            samples = [np.hstack([P, np.zeros((len(P), 1))]) for P in samples]
        # up = np.array([0, 0, penup_distance])
        # samples = ([np.array([samples[0][0]+up, samples[0][0]])]
        #                     + samples
        #                     + [np.array([samples[-1][-1], samples[-1][-1]+up])])
        paths, simplified = planning_callback(samples,
                                              penup_distance=penup_distance,
                                              max_speed=max_speed,
                                              accel_time=accel_time,
                                              params=params)
        paths = [P for P in paths if len(P) > 1]

        print('lens', [len(P) for P in paths])
        if not paths:
            return []
        trajectories = [paths[0]]
        if len(paths) > 1:
            for i in range(len(paths)-1):
                P1 = paths[i]
                P2 = paths[i+1]
                if geom.distance(P1[-1], P2[0]) > 2e-3:
                    trajectories.append(tu.lifting_motion(P1[-1], P2[0], penup_distance, max_accel_time=accel_time))
                    #distances_down.append(0.0)
                trajectories.append(P2)
                #distances_down.append(geom.chord_length(P2))

    else:
        # Create one motor plan with pen-ups and plan single trajectory
        trajectories, simplified = planning_callback(samples,
                                                     penup_distance=penup_distance,
                                                     max_speed=max_speed,
                                                     accel_time=accel_time,
                                                     params=params)
        #distances_down = [geom.chord_length(traj) for traj in trajectories]

    velocities = [np.diff(P, axis=0) / cfg.dt for P in trajectories]
    accelerations = [np.diff(P, n=2, axis=0) / (cfg.dt**2) for P in trajectories]
    max_accel = max_speed / accel_time

    if debug:
        ply.figure(title='Cartesian trajectory')

        Tv = 0.0
        Ta = 0.0

        for i, X in enumerate(trajectories):
            #speed = velocities[i] #tu.compute_speed(X, cfg.dt)
            #accel = accelerations[i] #tu.compute_accel(X, cfg.dt)
            tv = np.linspace(0, (len(velocities[i])-1)*cfg.dt, len(velocities[i])) + Tv
            ta = np.linspace(0, (len(accelerations[i])-1)*cfg.dt, len(accelerations[i])) + Ta
            ply.subplot(1, 1, title='Velocity')
            ply.plot(tv, velocities[i][:,0], color=ply.categorical_color(i))
            ply.plot(tv, velocities[i][:,1], color=ply.categorical_color(i))
            ply.plot(tv, velocities[i][:,2], color='b', alpha=0.5)
            ply.subplot(1, 2, title='Acceleration')
            ply.plot(ta, accelerations[i][:,0], color=ply.categorical_color(i))
            ply.plot(ta, accelerations[i][:,1], color=ply.categorical_color(i))
            ply.plot(ta, accelerations[i][:,2], color='b', alpha=0.5)
            Tv = tv[-1] + cfg.dt
            Ta = ta[-1] + cfg.dt

        ply.subplot(1, 1, title='Velocity')
        ply.rectangle([0, -max_speed], [Tv, max_speed], color='k', alpha=0.1, fill=True)
        ply.subplot(1, 2, title='Acceleration')
        ply.rectangle([0, -max_accel], [Ta, max_accel], color='k', alpha=0.1, fill=True)


        ply.subplot(2, 1, title='Trajectory 2d')
        ply.stroke(samples, 'k', alpha=0.25)
        ply.stroke(simplified, 'r', ls=':', lw=1)
        ply.stroke(trajectories, 'k')
        ply.axis_equal(True)

        ply.subplot(2, 2, title='Trajectory 3d')

        box = geom.bounding_box(trajectories)

        for X in trajectories:
            X = np.array(X)
            vals = (X[:,2])/(box[1][2] - box[0][2])
            ply.stroke3d(X, color=[ply.cmap(z, 'magma') if z >=0 else ply.cmap(-z, 'viridis') for z in vals])
        ply.show()
    return trajectories

def load(path, params):
    if '.svg' in path:
        return load_svg(path, params)
    else:
        return load_thick_paths(path)
        
def load_thick_paths(path):
    strokes = utils.load_json(path)
    if not strokes:
        return
    chains = [np.array(P) for P in strokes['points']]
    widths = [np.array(w) for w in strokes['stroke_widths']]
    return [np.hstack([P, w.reshape(-1,1)]) for P, w in zip(chains, widths)]

def load_svg(path, params):
    print('Loading SVG:', path)
    try:
        draft = svg.load_svg_bezier_chains(path)
    except FileNotFoundError as e:
        print(e)
        return
    zz = svg.load_svg_polylines(path)

    dest_rect = geom.make_rect(0, 0, 1024, 768) 
    draft_in = geom.transform_to_rect(draft, dest_rect)
    if params.sorted:
        draft_in = spatial.sort_paths(draft_in)
    draft = []
    tol = 0.2
    for chain in draft_in: #[2:4]:
        if params.bezier_split_angle > 0:
            chunks = bezier.split_chain(chain, angle_tol=params.bezier_split_angle)
        else:
            chunks = [chain]
        chunk_beziers = []
        for chunk in chunks:
            chunk_beziers.append(chunk)
            # pts = bezier.cubic_bezier_adaptive(chunk, tol)
            #pts = bezier.bezier_piecewise(chunk, 30)
            # beziers = fitCurve(pts, 4)
            # chunk_beziers += beziers #.append(beziers)

        for chain in chunk_beziers:
            #chain = bezier.beziers_to_chain(chunk_beziers)
            P = bezier.cubic_bezier_adaptive(chain, tol)
            #P = geom.adaptive_subdivide(P, 5)
            draft += [P] # geom.split_polyline_at_angle(P, 160)
        #draft.append(P)
        # print('chain len:', len(chain))
        # draft.append(bezier.beziers_to_chain(beziers))

    return draft
    