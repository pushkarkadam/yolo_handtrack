import sys 

sys.path.append('../..')

import seamdet as sd


data = sd.path.point_plotter('rectified_imageL.png', 'checkpoints', ["check_points"], ["x"], ["r"])

loaded_data = sd.path.load_checkpoint_gt('checkpoints.pkl')

try: 
    assert(data == loaded_data)
    print('\033[92m' + "Success!")
except:
    print('\033[91m' + "Loaded data does not match the return data.")