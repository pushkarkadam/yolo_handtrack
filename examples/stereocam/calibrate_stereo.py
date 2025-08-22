import sys 

sys.path.append('../')
from stereocam import *

def main():
    # executing the stereo_calibration
    stereo_calibration(file_path='../images/case2', pattern_size=(8, 4), chess_box_size=30, save_rendered="../images/case2_calib_rev")

if __name__ == '__main__':
    main()