from importlib import reload
from tiba import svg, geom, bezier, spatial, interface, config
import tiba.matplotlyb as ply
import tiba.trajectory_utils as tu
import numpy as np
import matplotlib.pyplot as plt
reload(ply)
reload(spatial)
reload(interface)

itiba = interface.TibaInterface()
cfg = config.cfg

params = lambda: None
params.bezier_split_angle = 110

from tiba import utils, lqt
import os
from py5canvas import canvas
from PIL import Image
reload(lqt)
reload(geom)
reload(interface)
reload(bezier)
reload(tu)

def det22(mat):
    return mat[0,0] * mat[1,1] - mat[0,1]*mat[1,0]

strokes = utils.load_json(os.path.expanduser('~/Dropbox/develop_box/py5sketches/tiba_tests/teststrokes2.json'))
chains = [np.array(P) for P in strokes['points']]
widths = [np.array(w) for w in strokes['stroke_widths']]
chains, mat = itiba.pad.fit_to_box(chains, padding=200, penup_mm=0.0, get_mat=True)
scale = np.sqrt(abs(det22(mat)))
widths = [w*scale for w in widths]
width_range_mm = [np.min(widths)*1000, np.max(widths)*1000]
print('wrange mm', width_range_mm)

def map(x, input_range, output_range):
    return (output_range[0] + ((x - input_range[0])/(input_range[1] - input_range[0])) 
                                * (output_range[1] - output_range[0]))

wrange = np.array(width_range_mm)/1000.0
zrange = np.array([0.0, -5.0])/1000.0

def width_to_z(w):
    return map(w, wrange, zrange)

def z_to_width(z):
    return map(z, zrange, wrange)

chains_3d = [np.hstack([Cp, width_to_z(w.reshape(-1,1))]) for Cp, w in zip(chains, widths)]
trajectories = [tu.plan_lqt_bezier(Cp, 7.5, cfg.cartesian_speed, cfg.acceleration_time, alpha=0.0, order=3) for Cp in chains_3d]
#trajectories = tu.plan_polylines_time_optimal(trajectories, cfg.cartesian_speed, cfg.acceleration_time)
c = canvas.Canvas(512, 512)
c.set_color_scale(1.0)
c.background(1)
c.apply_matrix(np.linalg.inv(mat))
for Cp, w in zip(chains, widths):
    print(len(Cp))
    S = bezier.thick_bezier_curve(Cp, w) #0.01)
    c.fill(1,0,0,0.9)
    c.no_stroke()
    c.shape(S)
for Xw in trajectories:
    S = geom.thick_curve(Xw, z_to_width)
    c.fill(0)
    c.no_stroke()
    c.shape(S)

c.show()

from tiba import draw
reload(draw)
reload(tu)
reload(ply)

penup_distance_mm = 20.0
trajectories_penup = [trajectories[0]]
if len(trajectories) > 1:
    for i in range(len(trajectories)-1):
        P1 = trajectories[i]
        P2 = trajectories[i+1]
        if geom.distance(P1[-1], P2[0]) > 1e-3:
            trajectories_penup.append(tu.lifting_motion(P1[-1], P2[0], penup_distance_mm/1000.0))
        trajectories_penup.append(P2)

draw.plot_cartesian_trajectory_stats(trajectories_penup, backend='plotly', figsize=(20, 10))

robot = itiba.robot
cur_pos = robot.pen_tcp_pos(robot.q)
trajectories_pad = itiba.pad.fit_to_frame(trajectories_penup)
trajectories_pad = itiba.pad.augment_with_pen_orientation(trajectories_pad, robot, robot.home)
trajectories_pad += [tu.trapezoidal_approach(trajectories_pad[-1][-1], cur_pos)] 
itiba.plan_task_trajectories(trajectories_pad)
print("Finished planning task trajectories")
itiba.plot_trajectory_stats(trajectories_pad) #self.task_trajectories_augmented)
itiba.simulate()
print("End")