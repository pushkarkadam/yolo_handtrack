import cv2
import os


def get_video_frames(video_path, image_format=None, resize=(1280, 720)):
    """Returns a list of image frames as numpy array.
    
    The frames are extracted from the video and each frame
    is an image that is represented as a numpy array.
    
    Parameters
    ----------
    video_path: str
        The path to the video file.
    image_format: int
        Use the ``enum`` from opencv to specify the conversion format.
        Most likely, the conversion will be from BGR to RGB.
        To perform this operation, use ``image_format = cv2.COLOR_BGR2RGB``
        
    Returns
    -------
    list
        A list of image frames in the video as numpy array.
    
    Examples
    --------
    >>> from lfdtrack.fingertrack import *
    >>> video_path = "~/path/to/video/file.mp4"
    >>> frames = get_video_frames(video_path)
    
    """
    # Empty list to store the frames
    frames = []
    
    video = cv2.VideoCapture(video_path)
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(total_frames):
        _, frame = video.read()
        # Appending the frame to the list
        frames.append(frame)
        
    if image_format:
        new_frames = []
        for frame in frames:
            new_frames.append(cv2.cvtColor(frame, image_format))

        return new_frames

    return frames

def convert_video_to_images(video_path, save_path):
    """Saves each frame of the video as an image
    
    Parameters
    ----------
    video_path: str
        Path to the location of the video.
    save_path: str
        Path to where the images need to be saved.

    """

    frames = get_video_frames(video_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_paths = [os.path.join(save_path, str(i) + ".png") for i in range(len(frames))]

    for frame, image_path in zip(frames, image_paths):
        cv2.imwrite(image_path, frame)
        
def reduce_fps(video_file, save_path, fps=3):
    """Reduces FPS of the video.
    
    Parameters
    ----------
    video_files: str
        Path to the video file.
    save_path: str
        Path where the images will be saved.
        ``save_path`` provides the parent directory.
        ``images`` directory will be created as a child and the images with ``.png`` format will be stored inside the child directory.
    fps: int, default ``3``
        Frames per second to convert the video.
        
    """
    vidcap = cv2.VideoCapture(video_file)
    assert vidcap.isOpened()

    frames = []

    fps_in = vidcap.get(cv2.CAP_PROP_FPS)

    index_in = -1
    index_out = -1

    while True:
        success = vidcap.grab()
        if not success:
            break
        index_in += 1

        out_due = int(index_in / fps_in * fps)

        if out_due > index_out:
            success, frame = vidcap.retrieve()
            if not success:
                break
            index_out += 1

            frames.append(frame)

    save_dir = os.path.join(save_path, "images")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"{save_dir} created.")
    else:
        print(f"{save_dir} already exists.")

    for idx, frame in enumerate(frames):
        cv2.imwrite(f"{save_dir}/{idx}.png", frame)