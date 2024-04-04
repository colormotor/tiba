#!/usr/bin/env python3
# NB THIS IS HAND
pick_pos = np.array([0.21363804, 0.45648885, 0.08957076])
up = np.array([0, 0, 1])

start_offset = 150/1000

joint_move('Tool side start')
print('Align hand')
hand_align_pick_tool()
goto_xyz_hand(pick_pos + up*start_offset)
goto_xyz_hand(pick_pos)
grasp()
goto_xyz_hand(pick_pos + up*start_offset)
joint_move('Home')
