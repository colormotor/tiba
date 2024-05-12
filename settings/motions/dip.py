#!/usr/bin/env python3
require_args("dip_radius_mm", "clean_radius_mm")

# Position for center of rim
dip_pos = np.array([0.11702348446934795, -0.4045200962012843, 0.1051926181034665])

up = np.array([0, 0, 1])

start_offset = 70/1000
dip_offset = -35/1000
clean_offset = -12/1000

joint_move('Home')
joint_move('Dip')
tool_align_dip()

# Move above cup
goto_xyz(dip_pos + up*start_offset)
# Move inside cup
goto_xyz(dip_pos + up*dip_offset)
# Dipping circles
xy_circles(dip_radius_mm, 4)
# # Move to rim pos
goto_xyz(dip_pos + up*clean_offset)
# # Cleaning circles
xy_circles(clean_radius_mm, 4)
goto_xyz(dip_pos + up*start_offset)
joint_move('Home')
