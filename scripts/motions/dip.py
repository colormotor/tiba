#!/usr/bin/env python3
require_args("dip_radius_mm", "clean_radius_mm")

dip_pos = np.array([0.11812825731443674, -0.407704314564537, 0.10439764358100054])
up = np.array([0, 0, 1])

start_offset = 70/1000
dip_offset = 15/1000
clean_offset = 30/1000

joint_move('Dip')
hand_align()
goto_xyz(dip_pos + up*start_offset),
goto_xyz(dip_pos + up*dip_offset),
xy_circles(dip_radius_mm, 4)
goto_xyz(dip_pos + up*clean_offset)
xy_circles(clean_radius_mm, 4)
goto_xyz(dip_pos + up*start_offset)
joint_move('Home')
