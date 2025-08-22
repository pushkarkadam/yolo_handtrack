import sys 
import time
import cv2

sys.path.append('../')
from stereocam import *

def main():
    # Calibration data path

    stereo_config = load_stereo_params("stereo_config.yaml")

    calib_path = stereo_config["calib_path"]
    imageL_path = stereo_config["imageL_path"]
    imageR_path = stereo_config["imageR_path"]

    # time calculation
    start = time.time()

    # loading calibration data
    calib_data = load_calibration_data(calib_path)

    # Reading images as BGR
    imageL = cv2.imread(imageL_path)
    imageR = cv2.imread(imageR_path)

    # converting gray images
    grayL, grayR = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in [imageL, imageR]]

    # Converting to HSV colorspace
    imgL_hsv, imgR_hsv = colorspace_transform(imageL, imageR)

    # filtering the images
    imgL_fil, imgR_fil = clahe_filter(imgL_hsv, imgR_hsv)

    # Stereo map generation
    stereoMapL, stereoMapR = stereo_map(calib_data, image_shape=grayL.shape[::-1])

    # Image rectification - hsv
    rectL, rectR = rectify_images(imgL_fil, imgR_fil, stereoMapL, stereoMapR)

    # image rectification - bgr
    bgrRectL, bgrRectR = rectify_images(imageL, imageR, stereoMapL, stereoMapR)

    # Depth maps generation
    disparity, camera_projection, depth_map, left_cut = depth_maps(imageL=rectL, imageR=rectR, Q=calib_data['Q'])

    end = time.time() - start

    print(f"Time elapsed: {end} s")

    # 3d point cloud data
    pcd = point_cloud(bgrRectL, depth_limits=(0, 0.5), camera_projection=camera_projection, depth_map=depth_map, left_cut=left_cut, image_type='bgr')

if __name__ == '__main__':
    main()