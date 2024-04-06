#!/usr/bin/env python3
require_args("dip_radius_mm", "clean_radius_mm")

# Position for center of rim
#dip_pos = np.array([0.11812825731443674, -0.407704314564537+3/1000, 0.10439764358100054])
#dip_pos = np.array([0.11746458629108344, -0.40536088024319067, 0.190187315063793])
dip_pos = np.array([0.11702348446934795, -0.4045200962012843, 0.1051926181034665])

up = np.array([0, 0, 1])

start_offset = 70/1000
dip_offset = -35/1000
clean_offset = -12/1000

joint_move('Dip')
tool_align_dip()

# Move above cup
goto_xyz(dip_pos + up*start_offset)
# Move inside cup
goto_xyz(dip_pos + up*dip_offset)
# Dipping circles
# print("Dip circles")
xy_circles(dip_radius_mm, 4)
# # Move to rim pos
# print("Moving up")
# goto_xyz(dip_pos + up*clean_offset)
# # Cleaning circles
# print("Rim circles")
# xy_circles(clean_radius_mm, 4)
# goto_xyz(dip_pos + up*start_offset)
# joint_move('Home')
