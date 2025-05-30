import cv2
import argparse
import datetime
import os


def capture_stereo(output_path="images", camera_number=None, width=4416, height=1242):
    """Captures stereo images for calibration.
    
    Parameters
    ----------
    output_path: str
        Path to store the video.
        The function will create two sub-directories ``stereo_left`` and ``stereo_right``.
    camera_number: int
        The camera number.
        This can be ``None`` if the camera is unknown the
        :func:`stereocam.capture_images.detect_camera`
        and
        :func:`stereocam.capture_images.detect_stereo`
        are used.
    width: int
        Width of the image.
        This is the total width of the stereo camera that includes left and right image.
        For getting ``1920 x 1080`` resolution, the width should be ``1920 x 2 = 3840``.
    height: int
        Height of the image.
        Both the cameras of the stereo rig will have same height.
    
    """
    

    # If the camera number index is not provided in the function then perform auto detection
    if not camera_number:
        # Detecting available cameras
        cam_available = detect_camera()

        # Selecting stereo camera
        _, camera_number = detect_stereo(cam_available)

    # date
    ct = datetime.datetime.now()
    date = ct.strftime("%d-%m-%Y-%H-%M")

    # Make directory for saving both left and right images
    path_left = os.path.join(output_path, date, 'stereo_left')
    path_right = os.path.join(output_path, date, 'stereo_right')

    print(f"Creating directories: \n{path_left}\n{path_right}")
    os.makedirs(path_left)
    os.makedirs(path_right)

    # create a video capture for stereo camera
    cap = cv2.VideoCapture(camera_number)

    # Creating resolution
    resolution = (int(width), int(height))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Performing a five interation count to ensure that the camera starts correctly
    # This is very important because in the first instance, the camera may not be ready
    # If this step is not performed, then a green output is expected in the image.
    count_till = 5

    print("Initialising camera...")
    for i in range(count_till):
        ret, frame = cap.read()
        print(f"Frame count: {i}")

        if not ret:
            print('\033[91m' + "Camera failed to start!")
            break

    num_images = 0

    while cap.isOpened():
        # Read frames from the stereo camera
        retval, frame = cap.read()

        # Checking if retval (return value) is false and if so, then it stops the code
        if not retval:
            print('\033[91m' + "Camera failed to start!")
            break

        # Split the frame into left and right images
        left_frame = frame[:, :width // 2, :]
        right_frame = frame[:, width // 2:, :]

        # Waiting for 5 milliseconds
        key = cv2.waitKey(5)

        # Check if ESC key is pressed whose ASCII value is 27
        if key == 27:
            break 
        elif key == ord('s'):
            cv2.imwrite(os.path.join(path_left, 'imageL_' + str(num_images) + '.png'), left_frame)
            cv2.imwrite(os.path.join(path_right, 'imageR_' + str(num_images) + '.png') , right_frame)
            print('\033[92m' + "Images saved!")
            num_images += 1

        cv2.imshow("Left stereo", left_frame)
        cv2.imshow("Right stereo", right_frame)


    # Release the video capture and video writers
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()


def detect_camera(max_cam=10):
    """Detects camera"""

    cam_available = []

    for cam in range(max_cam):
        try:
            cap = cv2.VideoCapture(cam)
        except Exception as e:
            print(e)
        
        ret, _ = cap.read()

        if ret:
            cam_available.append(cam)

        cap.release()
        cv2.destroyAllWindows()

    return cam_available

def detect_stereo(cam_available, stereo_res=(1242, 4416)):
    """Returns the index of stereo camera.

    Making certain assumptions about what stereo images consist.
    Assuming the stereo camera is connected using a USB and USB is the last camera to be detected.
    Assuming stereo camera has a certain resolution that is expected.

    Parameters
    ----------
    cam_available: int
        Checks if the system camera input is a stereo.
    stereo_res: tuple, default ``(1242, 4416)``
        Resolution of the left and right combined images of the stereo camera.
    
    """
    # Using the last value of the cam_available list because it has a potential to be stereo
    stereo_index = cam_available.pop()

    stereo_retval = True

    cap = cv2.VideoCapture(stereo_index)

    # Creating resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_res[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stereo_res[0])

    # Getting image
    ret, frame = cap.read()
    
    if frame.shape[:-1] == stereo_res:
        print('\033[92m' + 'Stereo possible')
    else:
        print('\033[93m' + 'May not be stereo. Please check manually and use correct index!')
        stereo_retval = False

    cap.release()
    cv2.destroyAllWindows()

    return stereo_retval, stereo_index

def initialise_camera(cap):
    """Initialises the camera for five iterations to avoid missing data.

    Parameters
    ----------
    cap: cv2.VideoCapture
        VideoCapture object of opencv
    
    """
    count_till = 5

    print("Initialising camera...")
    for i in range(count_till):
        ret, frame = cap.read()
        print(f"Frame count: {i}")

        if not ret:
            print('\033[91m' + "Camera failed to start!")
            break

def record_stereo(output_path, camera_number, width, height, fps=10, save_images=False):
    """Records stereo videos.
    
    Parameters
    ----------
    output_path: str
        Path to store the video.
        The function will create two sub-directories ``stereo_left`` and ``stereo_right``.
    camera_number: int
        The camera number.
        This can be ``None`` if the camera is unknown the
        :func:`stereocam.capture_images.detect_camera`
        and
        :func:`stereocam.capture_images.detect_stereo`
        are used.
    width: int
        Width of the image.
        This is the total width of the stereo camera that includes left and right image.
        For getting ``1920 x 1080`` resolution, the width should be ``1920 x 2 = 3840``.
    height: int
        Height of the image.
        Both the cameras of the stereo rig will have same height.

    """

    # If the camera number index is not provided in the function then perform auto detection
    if not camera_number:
        # Detecting available cameras
        cam_available = detect_camera()

        # Selecting stereo camera
        _, camera_number = detect_stereo(cam_available)

    # date 
    ct = datetime.datetime.now()
    date = ct.strftime("%d-%m-%Y-%H-%M")

    # Make directory for saving both left and right images
    path_left = os.path.join(output_path, date, 'stereo_left')
    path_right = os.path.join(output_path, date, 'stereo_right')

    print(f"Creating directories: \n{path_left}\n{path_right}")
    os.makedirs(path_left)
    os.makedirs(path_right)

    # Open the video capture for stereo camera
    cap = cv2.VideoCapture(camera_number)

    # Creating resolution
    resolution = (int(width), int(height))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # wait till the camera has been initialised for 5 iterations
    initialise_camera(cap)

    # Create video writers for left and right cameras
    left_writer = cv2.VideoWriter(os.path.join(path_left, 'left.avi'), cv2.VideoWriter_fourcc(*'XVID'), fps, (width // 2, height))
    right_writer = cv2.VideoWriter(os.path.join(path_right, 'right.avi'), cv2.VideoWriter_fourcc(*'XVID'), fps, (width // 2, height))

    
    # Creates "images" directory inside the stereo left and right directory
    if save_images:
        left_images_path = os.path.join(path_left, "images")
        right_images_path = os.path.join(path_right, "images")

        if not os.path.exists(left_images_path):
            os.makedirs(left_images_path)
        if not os.path.exists(right_images_path):
            os.makedirs(right_images_path)

    # Counts the frame
    frame_counter = 0

    while cap.isOpened():
        # Read frames from the stereo camera
        retval, frame = cap.read()

        if not retval:
            print('\033[91m' + "Camera failed to start!")
            break

        # Split the frame into left and right images
        left_frame = frame[:, :width // 2, :]
        right_frame = frame[:, width // 2:, :]

        if save_images:
            left_image_save_path = os.path.join(left_images_path, str(frame_counter) + ".png")
            right_image_save_path = os.path.join(right_images_path, str(frame_counter) + ".png")

            cv2.imwrite(left_image_save_path, left_frame)
            cv2.imwrite(right_image_save_path, right_frame)

        # Write frames to the corresponding video writers
        left_writer.write(left_frame)
        right_writer.write(right_frame)

        # # Display the frames
        cv2.imshow('Left Camera', left_frame)
        cv2.imshow('Right Camera', right_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_counter += 1

    # Release the video capture and video writers
    cap.release()
    left_writer.release()
    right_writer.release()

    # Destroy all windows
    cv2.destroyAllWindows()

def split_stereo(frame):
    """Splits the stereo image into left and right frames.

    Parameters
    ----------
    frame: numpy.ndarray
        Stereo image frame.

    """
    height, width, _ = frame.shape 

    left_frame = frame[:, :width // 2, :]
    right_frame = frame[:, width // 2:, :]

    return left_frame, right_frame