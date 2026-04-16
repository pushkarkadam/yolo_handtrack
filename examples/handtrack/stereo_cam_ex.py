import cv2
import sys 

sys.path.append('../..')

import stereocam as sc


cam_index = sc.capture_images.detect_stereo_camera("zed")

cap = cv2.VideoCapture(cam_index)

print("Press 'q' to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break


    cv2.imshow('Camera Feed', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()