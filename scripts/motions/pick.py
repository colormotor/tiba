#!/usr/bin/env python3
pick_pos = np.array([0.21385224976244546, 0.4576594453496983, -0.0035341840649663336])
up = np.array([0, 0, 1])

start_offset = 150/1000

joint_move('Tool side start')
print('Align hand')
hand_align_pick_tool()
goto_xyz(pick_pos + up*start_offset)
goto_xyz(pick_pos)
grasp()
goto_xyz(pick_pos + up*start_offset)
# joint_move('Tool')
joint_move('Home')
