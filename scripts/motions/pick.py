#!/usr/bin/env python3
pick_pos = np.array([0.1795643228310041, 0.407352498054852, -0.027149068876857835])
up = np.array([0, 0, 1])

start_offset = 100/1000

joint_move('Tool')
hand_align()
goto_xyz(pick_pos + up*start_offset)
goto_xyz(pick_pos)
grasp()
goto_xyz(pick_pos + up*start_offset)
joint_move('Tool')
joint_move('Home')
