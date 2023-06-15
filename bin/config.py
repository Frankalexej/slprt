# project root
ROOT = "../"
# source directories
SRC_DIR = ROOT + "src/"

# mediapipe model (.task)
MP_MODEL_PATH = SRC_DIR + "mediapipe/hand_landmarker.task"
# trial images
IMG_DIR = SRC_DIR + "img/"

# source video dir
vid_dir = SRC_DIR + "vid/"
vid_vid_dir = vid_dir + "vid/"
vid_pic_dir = vid_dir + "pic/"

# destination dirs: graph in det_dir, rendered videos and pics in rend_dir
det_dir = SRC_DIR + "det/"
rend_dir = SRC_DIR + "rend/"
rend_vid_dir = rend_dir + "vid/"
rend_pic_dir = rend_dir + "pic/"



# Constant Configs
VID_SUF = ".mp4"
S = "/"

# common frame size in the videos
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720