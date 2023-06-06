# import needed libs
import cv2
import math
import numpy as np
import os

import mediapipe as mp

from config import *
from mio import *
from draw_landmarks_on_image import *


# prepare mp
mp_hands = mp.solutions.hands
# Import drawing_utils and drawing_styles.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class DetectionWrapper: 
    def __init__(self, ptcp_file, filename): 
        self.name = filename

        self.vidpath = vid_vid_dir + ptcp_file + S + filename + VID_SUF
        self.dd = det_dir + ptcp_file + S + filename + S
        self.rpd = rend_pic_dir + ptcp_file + S + filename + S
        self.rvd = rend_vid_dir + ptcp_file + S

        # create specific det and rend dir (det dir is for detection result source file, rend dir is for rendered pics and vids)
        if not os.path.exists(self.dd): 
            os.makedirs(self.dd)
        if not os.path.exists(self.rpd): 
            os.makedirs(self.rpd)
        if not os.path.exists(self.rvd): 
            os.makedirs(self.rvd)

        self.images = []
        # self.annotated_images = []
        self.fps = 0    # to be obtained for video saving
        self.framesize = (FRAME_WIDTH, FRAME_HEIGHT)

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            self.FPS_TYPE = cv2.cv.CV_CAP_PROP_FPS
        else:
            self.FPS_TYPE = cv2.CAP_PROP_FPS
        

        # create detector
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO)
        self.detector = HandLandmarker.create_from_options(options)

        
    def getImgs(self): 
        # Path to video file
        vidObj = cv2.VideoCapture(self.vidpath)
        self.fps = vidObj.get(self.FPS_TYPE)

        # Used as counter variable
        count = 0  
        # checks whether frames were extracted
        success = 1
    
        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()
            # Saves the frames with frame-count
            if success: 
                self.images.append(image)
                count += 1
    
    def f2ms(self, idx): 
        return int(((idx + 1) / self.fps) * 1000)    # turn second to ms
    
    # def resize(self): 
    #     # relinquit
    #     for image in self.images:
    #         h, w = image.shape[:2]
    #         if h < w:
    #             img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    #         else:
    #             img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    
    def detect(self): 
        outvid = cv2.VideoWriter("{}{}.mp4".format(self.rvd, self.name),cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.framesize)
        print("Video name {}".format(self.name))

        for idx, image in enumerate(self.images):
            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            # results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect_for_video(mp_image, self.f2ms(idx))

            # ResultIO.save("{}{}_{:0>6}.pkl".format(self.dd, self.name, idx), results)

            # Draw pose landmarks and save to pic.
            # annotated_image = cv2.flip(image.copy(), 1)
            annotated_image = draw_landmarks_on_image(image, results)
            # for hand_landmarks in results.multi_hand_landmarks:
            #     # Print index finger tip coordinates.
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())
            # write image to image save dir
            cv2.imwrite("{}{}_{:0>6}.jpg".format(self.rpd, self.name, idx), cv2.flip(annotated_image, 1))
            # self.annotated_images.append(annotated_image)
            outvid.write(annotated_image)
        outvid.release()