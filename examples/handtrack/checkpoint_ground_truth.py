import sys 

sys.path.append('../..')

import seamdet as sd


data = sd.path.point_plotter('rectified_imageL.png', 'checkpoints', ["check_points"], ["x"], ["r"])