import mediapipe as mp


# CONFIG
VID_SUF = ".mp4"
S = "/"

# relinquit
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# common frame size in the videos
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720




# mediapipe consts
# prepare mp
mp_hands = mp.solutions.hands
# Import drawing_utils and drawing_styles.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

el = [list(element) for element in list(mp_hands.HAND_CONNECTIONS)]





FLAG_OK = 0
FLAG_NONE = 1
FLAG_FILLED = 2