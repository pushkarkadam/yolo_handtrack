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
        cv2.imwrite(image_path, frames)
        
