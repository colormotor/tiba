#!/usr/bin/env python3
require_args("T_tool", "T_approach")

up = np.array([0, 0, 1])

# SE3 -> Translation (Meter)
p_tool = np.array(T_tool.t)
p_approach = np.array(T_approach.t)


# Move inside cup
goto_xyz(p_tool)

# Move back above cup
goto_xyz(p_approach)