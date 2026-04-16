from ultralytics import YOLO
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import time
import csv


class YOLOHandPose:
    """Renders and extracts the landmarks.
    
    Parameters
    ----------
    frames: list, default ``[]``
        A list of ``numpy.ndarray`` of images.
    model_path: str, default ``'../models/freiHand0.pt'``
        File path for weights of the model.
    confidence_threshold: float, default 0.7
        Confidence threshold for rendering
        
    Attributes
    ----------
    results: list
        Store the results of detection.
    keypoints: list
        A list of 21 keypoints detected in image frame.
    xyn: list
        A list of normalised keypoint coordinates.
    xy: list
        A list of keypoint coordinates.
    EDGES: list
        A list of edges that connect the keypoints.
    rendered_images: list
        A list of rendered images with keypoint graph.
    model: ultralytics.yolo.engine.model.YOLO
        YOLO pose model.
    boxes: list
        A list of boxes from YOLO pose model.
    boxes_xywh: list
        A list of YOLO box coordinates ``[x y w h]``
    boxes_xywhn: list
        A list of normalised YOLO box coordinates ``[x y w h]``
    boxes_xyxy: list
        A list of YOLO box end vertices of box ``[x0 y0 x1 y1]``
    boxes_xyxyn: list
        A list of normalised YOLO box end vertices of box ``[x0 y0 x1 y1]``
    confidence: list
        Confidence score of detection.
    detections: list
        Class of detection
        
    Methods
    -------
    process()
        Process the video or image.
    render_pose(font_color=(0, 0, 0), edge_color=(255, 255, 255), landmark_color=(255, 0, 0),font_scale=0.2)
        Renders the pose and saves the rendered images in ``rendered_images``.
    save()
        Save image/video depending on the number of images in ``frames``.
    
    """
    def __init__(self, frames=[], model_path='../models/freiHand0.pt', confidence_threshold=0.7, **yolo_kw):
        self.frames = frames
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.results = []
        self.keypoints = []
        self.xyn = []
        self.xy = []
        self.EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
        self.rendered_images = []
        self.boxes = []
        self.boxes_xywh = []
        self.boxes_xywhn = []
        self.boxes_xyxy = []
        self.boxes_xyxyn = []
        self.confidence = []
        self.detections = []
        self.class_map = None
        self.yolo_kw = yolo_kw
        
        # Importing weights using YOLO
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(e)
                     
    def _extract_results(self, frame):
        """Extracts results.
        
        Parameters
        ----------
        frame: numpy.ndarray
            An image matrix.
            
        """
        # Using YOLO model to predict
        result = self.model(frame, **self.yolo_kw)
        
        # Appending results
        self.results.append(result)

        # Appending keypoints
        kpts = result[0].keypoints
        self.keypoints.append(kpts)
        
        # Appending boxes
        boxes = result[0].boxes
        self.boxes.append(boxes)
        
        # Adding confidence
        self.confidence.append(boxes.conf.cpu().numpy())
        
        # Adding class
        self.detections.append(boxes.cls.cpu().numpy())
        
        # Extracting class names
        self.class_map = result[0].names
        
        # Appending anchor coordinates detected in the image frame to a list
        self.boxes_xywh.append(result[0].boxes.xywh.cpu().numpy())
        self.boxes_xywhn.append(result[0].boxes.xywhn.cpu().numpy())

        # Appending box vertices to the list
        self.boxes_xyxy.append(result[0].boxes.xyxy.cpu().numpy())
        self.boxes_xyxyn.append(result[0].boxes.xyxyn.cpu().numpy())

        # Normalised keypoints
        xyn_array = result[0].keypoints.xyn.cpu().numpy()
        xyn_temp = []
        for i in xyn_array:
            xyn = [tuple(j) for j in i]
            xyn_temp.append(xyn)
        self.xyn.append(xyn_temp)

        # Keypoints
        xy_array = result[0].keypoints.xy.cpu().numpy()
        xy_temp = []
        for i in xy_array:
            xy = [tuple(j) for j in i]
            xy_temp.append(xy)
        self.xy.append(xy_temp)
    
    def process(self):
        """Processes the video frames.""" 
        for frame in self.frames:
            self._extract_results(frame)
            
    def render_pose(self, 
                    font_color=(0, 0, 0),
                    label_font_color=(255, 255, 255),
                    label_font_scale=0.8,
                    label_font_thickness=2,
                    edge_color=(255, 255, 255), 
                    landmark_color=(255, 0, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_thickness=2,
                    box_color=(255, 0, 0),
                    box_thickness=2,
                    font_scale=0.2,
                    show_landmarks=True,
                    show_box=True,
                    show_label=True
                   ):
        """Renders the image.
        
        Renders the bounding box, hand pose, class label and confidence.
        Also passes un-rendered images to ``self.rendered_images``. This helps
        in maintaining the continutity of input frames so that the rendered frames
        are same as input frames.
        
        Parameters
        ----------
        font_color: tuple, default ``(0, 0, 0)``
            Font color for landmark text.
        label_font_color: tuple, default ``(255, 255, 255)``
            Font color for label
        label_font_scale: float, default ``0.8``
            Font scale for label
        label_font_thickness: int, default ``2``
            Font thickness for label.
        edge_color: tuple, default ``(255, 255, 255)`` 
            Edge color
        landmark_color: tuple, default ``(255, 0, 0)``
            Landmark color
        font: int, default ``cv2.FONT_HERSHEY_SIMPLEX``
            Font
        font_thickness: int, default ``2``
            Font thickness for landmarks.
        box_color: tuple, default ``(255, 0, 0)``
            Color of bounding box.
        box_thickness: int, default ``2``
            Thickness of bounding box.
        font_scale: float, default ``0.2``
            Font scale for landmarks.
        show_landmarks: bool, default ``True``
            Renders landmarks
        show_box: bool, default ``True``
            Renders bounding box
        show_label: bool, default ``True``
            Shows the label with confidence and detection class.
            
        """
        for idx, image_frame in enumerate(self.frames):
            # Creating a deep copy
            frame = copy.deepcopy(image_frame)
            
            # For landmark
            if not self.xy[idx][0]:
                self.rendered_images.append(frame)
                continue
                
            for xy, xyxy, conf, det in zip(self.xy[idx], self.boxes_xyxy[idx], self.confidence[idx], self.detections[idx]):
                # Checking confidence treshold
                if conf > self.confidence_threshold:
                    if show_landmarks:
                        uv = [(np.int32(i[0]), np.int32(i[1])) for i in xy]
                        for e in self.EDGES:
                            frame = cv2.line(frame, uv[e[0]], uv[e[1]], edge_color, 2)

                        for n, landmark in enumerate(uv):
                            frame = cv2.circle(frame, landmark, 2, landmark_color, -1)
                            frame = cv2.putText(frame, 
                                                text=str(n), 
                                                org=landmark, 
                                                fontFace=font,
                                                fontScale=font_scale,
                                                color=font_color,
                                                thickness=font_thickness
                                               )
                    if show_box:
                        # Unpacking
                        x0, y0, x1, y1 = xyxy

                        start_point = (int(x0), int(y0))
                        end_point = (int(x1), int(y1))

                        # Bounding box
                        frame = cv2.rectangle(frame, start_point, end_point, box_color, box_thickness)

                        if show_label:
                            text = str(f"{self.class_map[det]}:{conf:.2f}")
                            text_size, _ = cv2.getTextSize(text, font, label_font_scale, font_thickness)
                            text_w, text_h = text_size
                            text_end_point = (start_point[0] + text_w, start_point[1] + text_h)
                            frame = cv2.rectangle(frame, start_point, text_end_point , box_color, -1)
                            frame = cv2.putText(frame,
                                                text=text,
                                                org=(start_point[0], start_point[1]+int(text_h)),
                                                fontFace=font,
                                                fontScale=label_font_scale,
                                                color=label_font_color,
                                                thickness=label_font_thickness
                                               )

            self.rendered_images.append(frame)
            
    def save(self, filename='rendered', path=".", frame_rate=30):
        """Saves the rendered images or video.
        
        Parameters
        ----------
        filename: str, default ``'rendered'``
            Name of the file to save the video.
        path: str, default ``'.'``
            Path where the vide with ``filename`` will be stored.
        frame_rate: int, default ``30``
            Frames per second in the video.
        
        """
        
        height, width, _ = self.rendered_images[0].shape
        
        file_path = os.path.join(path, filename)
        
        if len(self.frames) > 1:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec (XVID is a common codec)
            out = cv2.VideoWriter(file_path + '.avi', fourcc, frame_rate, (width, height))

            for image in self.rendered_images:
                out.write(image)

            out.release()
            
        else:
            cv2.imwrite(file_path + '.png', self.rendered_images[0])

class YOLOHandPoseLive(YOLOHandPose):
    """Live detection class

    Notes
    -----
        Larger network and large image size will create a lag in live detection.
        Compromising the accuracy by using nano network will lead to fast real time
        detection.
    
    Parameters
    ----------
    cam: int
        Camera number
    fps: int
        Frames per second
    model_path: str
        Path to the YOLO trained model.
    stereo_frame: str
        Select from three options: ``'left'``, ``'right'``, or ``''``.
        If empty string is provided as input, then it will use monocular camera.
    frame_size: tuple, default ``(1920, 1080)``
        Size of the frame.
        For stereo camera, either of the camera will be used for detection.
        Provide the resolution of the single camera and not about the two images combined.
        When using the integrated web cam of a laptop, make sure ``stereo_frame=''``.
        The empty string will use the monocular camera.
    confidence_threshold: float, default ``0.2``
        Confidence threshold in detection.
    **yolo_kw: dict
        Keyword arguments for YOLO model function.

    Methods
    -------
    stream()
        Detects pose in live.
        
    """

    def __init__(self, cam, fps, model_path, stereo_frame='left', frame_size=(1920, 1080), confidence_threshold=0.2, **yolo_kw):
        super().__init__(frames=[], model_path=model_path, confidence_threshold=confidence_threshold,**yolo_kw)
        self.cam = cam
        self.fps = fps
        self.frame_size = frame_size
        self.stereo_frame = stereo_frame

    def _reset(self):
        """Resets all the list to an empy list"""
        self.results = []
        self.xy = []
        self.rendered_images = []
        self.boxes_xyxy = []
        self.confidence = []
        self.detections = []

    def _extract_live_results(self, frame):
        """Same as _extract live function from parent class except
        that this function does not store information that is not needed 
        for rendering.

        Notes
        -----
            This function is optimised for real time detection only.
            The optimisation is performed by eliminating the processing of
            the unused variables during rendering process.

        Parameters
        ----------
        frame: numpy.ndarray
            Numpy image
        
        """
        # Using YOLO model to predict
        result = self.model(frame, **self.yolo_kw)
        
        # Appending results
        self.results.append(result)

        # Appending boxes
        boxes = result[0].boxes
        # self.boxes.append(boxes)
        
        # Adding confidence
        self.confidence.append(boxes.conf.cpu().numpy())
        
        # Adding class
        self.detections.append(boxes.cls.cpu().numpy())
        
        # Extracting class names
        self.class_map = result[0].names

        # Appending box vertices to the list
        self.boxes_xyxy.append(result[0].boxes.xyxy.cpu().numpy())

        # Keypoints
        xy_array = result[0].keypoints.xy.cpu().numpy()
        xy_temp = []
        for i in xy_array:
            xy = [tuple(j) for j in i]
            xy_temp.append(xy)
        self.xy.append(xy_temp)

    def _render_live_pose(self, 
                    image_frame,
                    font_color=(0, 0, 0),
                    label_font_color=(255, 255, 255),
                    label_font_scale=0.8,
                    label_font_thickness=2,
                    edge_color=(255, 255, 255), 
                    landmark_color=(255, 0, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_thickness=2,
                    box_color=(255, 0, 0),
                    box_thickness=2,
                    font_scale=0.2,
                    show_landmarks=True,
                    show_box=True,
                    show_label=True
                   ):
        """Renders the image.

        Notes
        -----
            This function is similar to ``render_pose()`` of the parent class.
            This function is optimized to ensure real time performance.
        
        Renders the bounding box, hand pose, class label and confidence.
        Also passes un-rendered images to ``self.rendered_images``. This helps
        in maintaining the continutity of input frames so that the rendered frames
        are same as input frames.
        
        Parameters
        ----------
        image_frame: np.ndarray
            A numpy image
        font_color: tuple, default ``(0, 0, 0)``
            Font color for landmark text.
        label_font_color: tuple, default ``(255, 255, 255)``
            Font color for label
        label_font_scale: float, default ``0.8``
            Font scale for label
        label_font_thickness: int, default ``2``
            Font thickness for label.
        edge_color: tuple, default ``(255, 255, 255)`` 
            Edge color
        landmark_color: tuple, default ``(255, 0, 0)``
            Landmark color
        font: int, default ``cv2.FONT_HERSHEY_SIMPLEX``
            Font
        font_thickness: int, default ``2``
            Font thickness for landmarks.
        box_color: tuple, default ``(255, 0, 0)``
            Color of bounding box.
        box_thickness: int, default ``2``
            Thickness of bounding box.
        font_scale: float, default ``0.2``
            Font scale for landmarks.
        show_landmarks: bool, default ``True``
            Renders landmarks
        show_box: bool, default ``True``
            Renders bounding box
        show_label: bool, default ``True``
            Shows the label with confidence and detection class.
            
        """
        # Index always remains zero since we have only one image to deal at a time
        idx = 0
        
        # Creating a deep copy
        frame = copy.deepcopy(image_frame)
        
        # For landmark
        if not self.xy[idx][0]:
            self.rendered_images.append(frame)
            return None
            
        for xy, xyxy, conf, det in zip(self.xy[idx], self.boxes_xyxy[idx], self.confidence[idx], self.detections[idx]):
            # Checking confidence treshold
            if conf > self.confidence_threshold:
                if show_landmarks:
                    uv = [(np.int32(i[0]), np.int32(i[1])) for i in xy]
                    for e in self.EDGES:
                        frame = cv2.line(frame, uv[e[0]], uv[e[1]], edge_color, 2)

                    for n, landmark in enumerate(uv):
                        frame = cv2.circle(frame, landmark, 2, landmark_color, -1)
                        frame = cv2.putText(frame, 
                                            text=str(n), 
                                            org=landmark, 
                                            fontFace=font,
                                            fontScale=font_scale,
                                            color=font_color,
                                            thickness=font_thickness
                                           )
                if show_box:
                    # Unpacking
                    x0, y0, x1, y1 = xyxy

                    start_point = (int(x0), int(y0))
                    end_point = (int(x1), int(y1))

                    # Bounding box
                    frame = cv2.rectangle(frame, start_point, end_point, box_color, box_thickness)

                    if show_label:
                        text = str(f"{self.class_map[det]}:{conf:.2f}")
                        text_size, _ = cv2.getTextSize(text, font, label_font_scale, font_thickness)
                        text_w, text_h = text_size
                        text_end_point = (start_point[0] + text_w, start_point[1] + text_h)
                        frame = cv2.rectangle(frame, start_point, text_end_point , box_color, -1)
                        frame = cv2.putText(frame,
                                            text=text,
                                            org=(start_point[0], start_point[1]+int(text_h)),
                                            fontFace=font,
                                            fontScale=label_font_scale,
                                            color=label_font_color,
                                            thickness=label_font_thickness
                                           )

        self.rendered_images.append(frame)

    def stream(self, **render_kw):
        """Detects pose in live.
        
        Parameters
        ----------
        render_kw: dict
            Keyword arguments that are the parameters of ``render_live_pose()`` methods.
            
        """

        width = self.frame_size[0]
        height = self.frame_size[1]
        
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Setting up 2x width for stereo camera or else width for monocular camera
        if self.stereo_frame:
            stereo_width = width * 2
        else:
            stereo_width = width

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        while True:
            # Starts the camera
            success, img = cap.read()

            # Split the frame into left and right images
            if not self.stereo_frame:
                img_name = 'monocular'
            elif self.stereo_frame == 'left':
                img = img[:, :width, :]
                img_name = 'left'
            elif self.stereo_frame == 'right':
                img = img[:, width:, :]
                img_name = 'right'

            self._extract_live_results(img)

            # Renders pose
            self._render_live_pose(img, **render_kw)

            # Extract the rendered image
            rendered_image = self.rendered_images[-1]

            # Display the image
            cv2.imshow(img_name, rendered_image)

            # reset all the populated list
            self._reset()

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class YOLOHandPoseLiveRecord(YOLOHandPose):
    "Live detection and record tracking class"

    def __init__(self, cam, fps, model_path, filename="", stereo_frame='left', frame_size=(1920, 1080), confidence_threshold=0.2, **yolo_kw):
        super().__init__(frames=[], model_path=model_path, confidence_threshold=confidence_threshold,**yolo_kw)
        self.cam = cam
        self.fps = fps
        self.frame_size = frame_size
        self.stereo_frame = stereo_frame
        self.filename = filename

        self.tracks = []
        self.tracks_timestamp = []
        self.first_frame = []

    def _reset(self):
        """Resets all the list to an empy list"""
        self.results = []
        self.xy = []
        self.rendered_images = []
        self.boxes_xyxy = []
        self.confidence = []
        self.detections = []

    def _extract_live_results(self, frame, index=8):
        """Same as _extract live function from parent class except
        that this function does not store information that is not needed 
        for rendering.

        Notes
        -----
            This function is optimised for real time detection only.
            The optimisation is performed by eliminating the processing of
            the unused variables during rendering process.

        Parameters
        ----------
        frame: numpy.ndarray
            Numpy image
        
        """
        # Using YOLO model to predict
        result = self.model.predict(frame, conf=self.confidence_threshold)
        
        # Appending results
        self.results.append(result)

        # Appending boxes
        boxes = result[0].boxes
        # self.boxes.append(boxes)
        
        # Adding confidence
        self.confidence.append(boxes.conf.cpu().numpy())
        
        # Adding class
        self.detections.append(boxes.cls.cpu().numpy())
        
        # Extracting class names
        self.class_map = result[0].names

        # Appending box vertices to the list
        self.boxes_xyxy.append(result[0].boxes.xyxy.cpu().numpy())

        xy_array = result[0].keypoints.xy.cpu().numpy()
        xy_temp = []
        for i in xy_array:
            xy = [tuple(j) for j in i]
            xy_temp.append(xy)
        self.xy.append(xy_temp)

    def _render_live_pose(self, 
                    image_frame,
                    font_color=(0, 0, 0),
                    label_font_color=(255, 255, 255),
                    label_font_scale=0.8,
                    label_font_thickness=2,
                    edge_color=(255, 255, 255), 
                    landmark_color=(255, 0, 0),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_thickness=2,
                    box_color=(255, 0, 0),
                    box_thickness=2,
                    font_scale=0.2,
                    show_landmarks=True,
                    show_box=True,
                    show_label=True,
                    show_tracks=True
                   ):
        """Renders the image.

        Notes
        -----
            This function is similar to ``render_pose()`` of the parent class.
            This function is optimized to ensure real time performance.
        
        Renders the bounding box, hand pose, class label and confidence.
        Also passes un-rendered images to ``self.rendered_images``. This helps
        in maintaining the continutity of input frames so that the rendered frames
        are same as input frames.
        
        Parameters
        ----------
        image_frame: np.ndarray
            A numpy image
        font_color: tuple, default ``(0, 0, 0)``
            Font color for landmark text.
        label_font_color: tuple, default ``(255, 255, 255)``
            Font color for label
        label_font_scale: float, default ``0.8``
            Font scale for label
        label_font_thickness: int, default ``2``
            Font thickness for label.
        edge_color: tuple, default ``(255, 255, 255)`` 
            Edge color
        landmark_color: tuple, default ``(255, 0, 0)``
            Landmark color
        font: int, default ``cv2.FONT_HERSHEY_SIMPLEX``
            Font
        font_thickness: int, default ``2``
            Font thickness for landmarks.
        box_color: tuple, default ``(255, 0, 0)``
            Color of bounding box.
        box_thickness: int, default ``2``
            Thickness of bounding box.
        font_scale: float, default ``0.2``
            Font scale for landmarks.
        show_landmarks: bool, default ``True``
            Renders landmarks
        show_box: bool, default ``True``
            Renders bounding box
        show_label: bool, default ``True``
            Shows the label with confidence and detection class.
            
        """
        # Index always remains zero since we have only one image to deal at a time
        idx = 0
        
        # Creating a deep copy
        frame = copy.deepcopy(image_frame)

        if show_tracks:
            tracks = copy.copy(self.tracks)
            if tracks:
                tracks_uv = [(np.int32(i[0]), np.int32(i[1])) for i in tracks]
    
                for n, kpt_uv in enumerate(tracks_uv):
                    frame = cv2.circle(frame, kpt_uv, 2, landmark_color, -1)
        
        # For landmark
        if not self.xy[idx][0]:
            self.rendered_images.append(frame)
            return None
            
        for xy, xyxy, conf, det in zip(self.xy[idx], self.boxes_xyxy[idx], self.confidence[idx], self.detections[idx]):
            # Checking confidence treshold
            if conf > self.confidence_threshold:
                if show_landmarks:
                    uv = [(np.int32(i[0]), np.int32(i[1])) for i in xy]
                    for e in self.EDGES:
                        frame = cv2.line(frame, uv[e[0]], uv[e[1]], edge_color, 2)

                    for n, landmark in enumerate(uv):
                        frame = cv2.circle(frame, landmark, 2, landmark_color, -1)
                        frame = cv2.putText(frame, 
                                            text=str(n), 
                                            org=landmark, 
                                            fontFace=font,
                                            fontScale=font_scale,
                                            color=font_color,
                                            thickness=font_thickness
                                           )
                if show_box:
                    # Unpacking
                    x0, y0, x1, y1 = xyxy

                    start_point = (int(x0), int(y0))
                    end_point = (int(x1), int(y1))

                    # Bounding box
                    frame = cv2.rectangle(frame, start_point, end_point, box_color, box_thickness)

                    if show_label:
                        text = str(f"{self.class_map[det]}:{conf:.2f}")
                        text_size, _ = cv2.getTextSize(text, font, label_font_scale, font_thickness)
                        text_w, text_h = text_size
                        text_end_point = (start_point[0] + text_w, start_point[1] + text_h)
                        frame = cv2.rectangle(frame, start_point, text_end_point , box_color, -1)
                        frame = cv2.putText(frame,
                                            text=text,
                                            org=(start_point[0], start_point[1]+int(text_h)),
                                            fontFace=font,
                                            fontScale=label_font_scale,
                                            color=label_font_color,
                                            thickness=label_font_thickness
                                           )


        self.rendered_images.append(frame)

    def _track_kpts(self, keypoint=8):
        """Trace keypoints provided.
        
        Parameters
        ----------
        keypoints: int
            Keypoint to track.
            
        """

        idx = 0

        print(f'confidence: {self.confidence[idx]}')

        if self.confidence[idx]:
            if self.confidence[idx][idx] >= self.confidence_threshold:
            
                # For landmark
                if self.xy[idx][idx]:

                    xy  = self.xy[idx][idx][keypoint]
            
                    self.tracks.append(xy)
                    self.tracks_timestamp.append(time.time())

    def _write_tracks(self):
        """Writes tracks to csv file."""
        
        header = ["timestamp", "x", "y"]
        
        with open(self.filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
    
            if header:
                if csvfile.tell() == 0:
                    csv_writer.writerow(header)

            for coords, timestamp in zip(self.tracks, self.tracks_timestamp):
                kpt_x, kpt_y = coords
                data_row = [timestamp, kpt_x, kpt_y]

                csv_writer.writerow(data_row)

    def _capture_first_frame(self):
        """Captures first frame for stereo processing"""

        width = self.frame_size[0]
        height = self.frame_size[1]
        
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Setting up 2x width for stereo camera or else width for monocular camera
        if self.stereo_frame:
            stereo_width = width * 2
        else:
            stereo_width = width

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        time.sleep(1)
        try:
            for i in range(5):
                print(f"Count: {i}")
                time.sleep(1)
                success, img = cap.read()
    
            if self.stereo_frame:
                # first frame
                img_left = img[:, :width, :]
                img_right = img[:, width:, :]
    
                self.first_frame.append(img_left)
                self.first_frame.append(img_right)
            else:
                self.first_frame.append(img)
        except Exception as e:
            print(e)
            cap.release()
            cv2.destroyAllWindows()

        cap.release()
        cv2.destroyAllWindows()

    def stream(self, **render_kw):
        """Detects pose in live.
        
        Parameters
        ----------
        render_kw: dict
            Keyword arguments that are the parameters of ``render_live_pose()`` methods.
            
        """

        self._capture_first_frame()

        width, height = self.frame_size
        
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Setting up 2x width for stereo camera or else width for monocular camera
        if self.stereo_frame:
            stereo_width = width * 2
        else:
            stereo_width = width

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        while True:
            try:
                # Starts the camera
                success, img = cap.read()
            
                # Split the frame into left and right images
                if not self.stereo_frame:
                    img_name = 'monocular'
                elif self.stereo_frame == 'left':
                    img = img[:, :width, :]
                    img_name = 'left'
                elif self.stereo_frame == 'right':
                    img = img[:, width:, :]
                    img_name = 'right'
    
                self._extract_live_results(img)
    
                # save tracks
                self._track_kpts()
    
                # Renders pose
                self._render_live_pose(img, **render_kw)

                # Extract the rendered image
                rendered_image = self.rendered_images[-1]
    
                # Display the image
                cv2.imshow(img_name, rendered_image)
    
                self._reset()
    
                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(e)
                cap.release()
                cv2.destroyAllWindows()
    
        cap.release()
        cv2.destroyAllWindows()

        self._write_tracks()