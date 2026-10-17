#!/usr/bin/env python3
require_args("dip_pos", "dip_depth_mm", "dip_radius_mm")

up = np.array([0, 0, 1])

start_offset = 70 / 1000
dip_offset = -dip_depth_mm / 1000

joint_move("Home")
joint_move("Dip")
tool_align_dip()

# Move above cup
goto_xyz(dip_pos + up * start_offset)
# Move inside cup
goto_xyz(dip_pos + up * dip_offset)
# Dipping circles
xy_circles(dip_radius_mm, 1)
goto_xyz(dip_pos + up * start_offset)
joint_move("Home")
