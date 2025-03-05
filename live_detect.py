import cv2
import mediapipe as mp
import os
from google.protobuf.json_format import MessageToDict
# import json
import numpy as np
from io import BytesIO
from PIL import Image


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

from model_dataset import FixedHandshapeDict
from paths import *
from model_model import LinearHandshapePredictor
from model_configs import *

# import PyQt5
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.dirname(PyQt5.__file__)


FLAG_NONE = 0
FLAG_OK = 1
FLAG_FILLED = 2 # set FLAG_FILLED = 1 if include interpolated data in interpolation next round

def lm_has_side_and_is_at(lm, side):
    # This is a renewed version of lm_has_side_and_is_at, considering the order is not strict
    # Use "Left", and "Right" for side, instead of numbers
    mh = lm.multi_handedness
    if mh is None: 
        return False, 0

    max_score = -1
    max_index = 0
    found = False
    for i, hand in enumerate(mh):
        handedness_dict = MessageToDict(hand)
        classification = handedness_dict["classification"][0]
        if classification["label"] == side:
            if classification["score"] > max_score:
                max_score = classification["score"]
                max_index = i
                found = True
    return found, max_index

def sol2json(d, json_path, side): 
    # with open(json_path, 'w') as fl:
    has, at = lm_has_side_and_is_at(d, side)
    if has: 
        ml = (MessageToDict(d.multi_hand_landmarks[at])["landmark"]) # 0 is one of the hands, do this first
        my_dict = {str(i): (d['x'], d['y'], d['z']) for i, d in enumerate(ml)}
        this_flag = FLAG_OK
    else: 
        my_dict = {str(i): (0, 0, 0) for i in range(21)} # default (0, 0, 0) for all nodes
        this_flag = FLAG_NONE
    # outdict = {"edges": el, "features": my_dict}  # for the current processing, it is not needed to include edges
    outdict = {"features": my_dict, "flag": this_flag}
    # fl.write(json.dumps(outdict, separators=(',', ':')))
    return outdict
    
def dict2array(dict_data):
    feature_list = [dict_data['features'][str(i)] for i in range(21)]
    feature_array = np.array(feature_list)
    return feature_array.reshape(1, 21, 3)


hsdict = FixedHandshapeDict()
def model_predict(model, features, hsdict): 
    this_features = torch.from_numpy(features)
    batch_num, lm_num, dim_num = this_features.size()
    x = this_features
    x = x.to(device)
    x = x.to(torch.float32)
    hid_rep, pred = model.predict(x, hsdict)
    hid_rep = hid_rep.cpu().detach().numpy()
    return hid_rep, pred
class ListBuffer: 
    def __init__(self, buffer_size) -> None:
        self.buffer = []
        self.buffer_size = buffer_size
    def append(self, item): 
        self.buffer.append(item)
        if len(self.buffer) > self.buffer_size: 
            self.buffer.pop(0)
    def get(self): 
        return self.buffer
    def stack_and_get(self): 
        return np.stack(self.buffer, axis=0)
class HandDetection:
    def __init__(self, video_path=None):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.handshape_dir = "./hs_char/"
        self.saved_frames_hidrep_right = ListBuffer(10)
        self.saved_frames_hidrep_left = ListBuffer(10)
        try_times = 10
        try_idxs = [-1, 0, 1, 2, -2]

        # Try opening the webcam or loading a video file
        for i in range(try_times): 
            for idx in try_idxs: 
                print(f"Trying to open the webcam (attempt {i+1}/{try_times})...")
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    break
            if self.cap.isOpened():
                    break
        if not self.cap.isOpened():
            print("No webcam found! Loading a video file...")
            if video_path and os.path.exists(video_path):
                self.cap = cv2.VideoCapture(video_path)
            else: 
                print("No video file found! Please provide a valid video file.")
                exit(1)
        # Set higher resolution for the camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Change to preferred width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Change to preferred height

        # Store the actual frame width and height
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def overlay_handshape(self, frame, prediction, position=(50, 50), size=(120, 120)):
        """Overlay predicted handshape image on the frame with a white background."""
        handshape_path = os.path.join(self.handshape_dir, f"{prediction}.png")
        if os.path.exists(handshape_path):
            handshape_img = cv2.imread(handshape_path, cv2.IMREAD_UNCHANGED)
            handshape_img = cv2.resize(handshape_img, size)

            # Ensure the image has an alpha channel (RGBA)
            if handshape_img.shape[2] == 4:
                overlay = handshape_img[:, :, :3]  # Extract RGB
                alpha = handshape_img[:, :, 3] / 255.0  # Normalize alpha
            else:
                overlay = handshape_img
                alpha = np.ones((size[1], size[0]))  # No transparency

            # Create a white background with a border
            white_bg = np.ones((size[1] + 10, size[0] + 10, 3), dtype=np.uint8) * 255
            x_offset, y_offset = 5, 5
            white_bg[y_offset:y_offset+size[1], x_offset:x_offset+size[0]] = overlay

            # Overlay the final image on the frame
            x, y = position
            h, w = white_bg.shape[:2]
            frame[y:y+h, x:x+w] = white_bg

        return frame

    def plot_hidden_representation(self, hidden_rep, size=(250, 150)):
        """Generate and return a small line plot of the hidden representation."""
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)

        num_dims = hidden_rep.shape[1]  # Assuming hidden_rep is (frames, dimensions)
        colors = plt.cm.viridis(np.linspace(0, 1, num_dims))  # Use a visually distinct colormap
        for dim, color in zip(range(num_dims), colors):
            ax.plot(hidden_rep[:, dim], color=color, linewidth=1.8, label=f'Dim {dim+1}')

        # ax.set_xlabel("Time (Frames)", fontsize=8, color="black")
        # ax.set_ylabel("Hidden Representation", fontsize=8, color="black")

        # Improve style for visibility
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.legend(fontsize=6, loc='upper right', frameon=False)  
        ax.set_facecolor("white")  
        fig.patch.set_facecolor("white")

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.resize(size)

        return np.array(img)

    def detect(self):
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            counter = 0
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or camera disconnected.")
                    break

                counter += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.flip(rgb_frame, 1)  # Mirror effect
                
                results = hands.process(rgb_frame)

                right_lm = sol2json(results, os.path.join("jsonsave", f"hand_landmarks_{counter}.json"), "Right")
                left_lm = sol2json(results, os.path.join("jsonsave", f"hand_landmarks_{counter}.json"), "Left")

                right_lm_array = dict2array(right_lm)
                left_lm_array = dict2array(left_lm)

                hid_rep_right, pred_right = model_predict(model, right_lm_array, hsdict)
                hid_rep_left, pred_left = model_predict(model, left_lm_array, hsdict)

                self.saved_frames_hidrep_right.append(hid_rep_right[0])
                self.saved_frames_hidrep_left.append(hid_rep_left[0])

                annotated_frame = cv2.flip(frame.copy(), 1)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                annotated_frame = self.overlay_handshape(annotated_frame, pred_right[0], position=(20, 50))
                annotated_frame = self.overlay_handshape(annotated_frame, pred_left[0], position=(20, 200))

                hidden_plot_size = (350, 150)
                hidden_plot_right = self.plot_hidden_representation(self.saved_frames_hidrep_right.stack_and_get(), size=hidden_plot_size)
                hidden_plot_left = self.plot_hidden_representation(self.saved_frames_hidrep_left.stack_and_get(), size=hidden_plot_size)

                annotated_frame[50:200, 900:1250] = cv2.resize(hidden_plot_right, hidden_plot_size)
                annotated_frame[220:370, 900:1250] = cv2.resize(hidden_plot_left, hidden_plot_size)

                # Display the frame
                cv2.imshow("Hand Detection with Overlays", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    model = LinearHandshapePredictor(
        input_dim=in_dim, 
        enc_lat_dims=enc_lat_dims, 
        hid_dim=hid_dim, 
        dec_lat_dims=dec_lat_dims, 
        output_dim=out_dim
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ts = "1113174414-lin"
    stop_epoch = "597"
    # save_subdir = os.path.join(model_save_dir, "{}/".format(ts))
    save_subdir = "./"
    model_raw_name = f"{stop_epoch}"
    model_name = model_raw_name + ".pt"
    model_path = os.path.join(save_subdir, model_name)
    state = torch.load(model_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    video_path = "./B_01_079-NOTHING-0KC7-67.mp4"
    detector = HandDetection(video_path=video_path)
    detector.detect()
