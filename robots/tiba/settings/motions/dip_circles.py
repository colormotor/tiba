#!/usr/bin/env python3
require_args("T_tool", "T_approach")

# SE3 -> Translation (Meter)
p_tool = np.array(T_tool.t)
p_approach = np.array(T_approach.t)


# Move inside cup
goto_xyz(p_tool)
xy_circles(4, 4)

# Move back above cup
goto_xyz(p_approach)