import os
import glob
import json
import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToDict
import plotly.express as px
import plotly.graph_objects as go
# from plotly.io import to_image
import matplotlib.pyplot as plt
from scipy.spatial import distance as d
from scipy.signal import savgol_filter
from configs import *

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
    with open(json_path, 'w') as fl:
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
        fl.write(json.dumps(outdict, separators=(',', ':')))


class GraphTool: 
    def __init__(self, data_dir, prefix):
        files = self._locate_files(data_dir, prefix)
        self.flag = np.array([])
        self.features = np.empty((0, 21, 3))    # 21, 3 is the dim of one frame
        for file_name in files: 
            with open(file_name, 'r') as file:
                data = json.load(file)
                
                # Extract flag value and append to the flags array
                flag_array = np.array([data['flag']])
                self.flag = np.concatenate((self.flag, flag_array))
                
                # Extract features and append to the features array
                feature_list = [data['features'][str(i)] for i in range(21)]
                feature_array = np.array(feature_list)
                self.features = np.concatenate((self.features, feature_array[None, :, :]))


    @staticmethod
    def _locate_files(data_dir, prefix):
        pattern = os.path.join(data_dir, prefix + '*')
        matching_files = glob.glob(pattern)
        
        # Sort the matching files alphabetically
        matching_files.sort()
        
        return matching_files

    @staticmethod
    def _find_closest(arr, target_index, target_value):
        if target_value not in arr: 
            return None
        # Prerequisite: at least one existing
        # Find the indices of elements with the target value
        indices_with_target_value = np.where(arr == target_value)[0]

        # Calculate the absolute differences between these indices and the target index
        differences = np.abs(indices_with_target_value - target_index)

        # Find the index with the smallest difference
        closest_index = indices_with_target_value[np.argmin(differences)]

        return closest_index

    # Function to perform sliding window linear interpolation
    def interpolate(self, window_size=3):
        """
        window_size: number of elements to consider on one side of the window
        """
        interpolated_features = np.copy(self.features)
        interpolated_flag = np.copy(self.flag)

        upper_bound = self.features.shape[0]

        # Iterate through each position in the flags array
        for i in range(upper_bound):
            # slide through the array; features and flag are of same length
            if interpolated_flag[i] == FLAG_NONE:   # if this frame is not filled
                start_idx = max(0, i - window_size) # either lower bound of array or edge of window
                end_idx = min(i + window_size + 1, upper_bound) # +1 because end_idx is not considered in slices

                window_flags = interpolated_flag[start_idx:end_idx]
                # Check if there are any OK features in the window
                if FLAG_OK in window_flags:
                    window_features = self.features[start_idx:end_idx]

                    # Find indices of OK features
                    ok_indices = np.where(window_flags == FLAG_OK)[0]

                    # Interpolate missing feature based on non-missing features
                    non_missing_features = window_features[ok_indices]
                    interpolated_feature = np.mean(non_missing_features, axis=0)

                    interpolated_features[i] = interpolated_feature
                else: 
                    if FLAG_OK not in interpolated_flag: 
                        # there is nothing to find, then use the old one
                        interpolated_features[i] = self.features[i]
                    else: 
                        # if there is any usable frame to interpolate, find the nearest as the transfer index
                        transfer_index = self._find_closest(interpolated_flag, i, FLAG_OK)
                        interpolated_features[i] = self.features[transfer_index]

                interpolated_flag[i] = FLAG_FILLED

        self.interpolated_features = interpolated_features
        self.interpolated_flag = interpolated_flag
        return

    def delete_empty(self): 
        # here: use FLAG_NONE or FLAG_FILLED to filter out all those that are not detected or have been interpolated and use this filter to clean the data. 
        # here we use the detection method same as in interpolation to work out which ones are missing. 

        # Create a boolean mask to identify items with FLAG_NONE
        mask = self.flag == FLAG_OK
        # Apply the mask to the features array
        filtered_features = self.features[mask]
        filtered_flag = self.flag[mask]
        self.flag = filtered_flag
        self.features = filtered_features
        # will replace the core data, instead of returning them 
    
    def get_features(self, flatten=False): 
        if flatten: 
            frame, lm, dim = self.features.shape
            return self.features.reshape((frame, lm * dim))
        else: 
            return self.features.copy()

class Smoother: 
    @staticmethod
    def moving_average(data, window_size=3):
        # Pad the data at the beginning and end to handle edge cases, it will copy the first element
        padded_data = np.pad(data, ((window_size, window_size), (0, 0), (0, 0)), mode='edge')

        # Calculate the moving average for each feature independently using convolution
        weights = np.ones(2 * window_size + 1) / (2 * window_size + 1)
        smoothed_data = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'), axis=0, arr=padded_data)

        return smoothed_data
    
    @staticmethod
    def exponential_moving_average(data, alpha=0.9):
        # Get the dimensions of the input data
        frame, feature, _ = data.shape

        # Create an empty array to store the smoothed data
        smoothed_data = np.zeros((frame, feature, 3))

        # Calculate the exponential moving average for each feature independently
        smoothed_data[:, :, :] = data[:, :, :]  # Copy the original data

        for t in range(1, frame):
            smoothed_data[t, :, :] = (1 - alpha) * data[t, :, :] + alpha * smoothed_data[t - 1, :, :]

        return smoothed_data
    
    @staticmethod
    def savitzky_golay_filter(data, window_size=5, polyorder=2):
        # Get the dimensions of the input data
        frame, feature, _ = data.shape

        # Reshape data for vectorized processing
        reshaped_data = data.reshape(frame * feature, 3)

        # Apply the Savitzky-Golay filter to the reshaped data
        smoothed_reshaped_data = np.apply_along_axis(lambda x: savgol_filter(x, window_size, polyorder), axis=0, arr=reshaped_data)

        # Reshape smoothed data back to original shape
        smoothed_data = smoothed_reshaped_data.reshape(frame, feature, 3)

        return smoothed_data


class GoodDict: 
    def __init__(self, thumb, index, middle, ring, pinky):
        self.THUMB = thumb
        self.INDEX = index
        self.MIDDLE = middle
        self.RING = ring
        self.PINKY = pinky

    def get_all(self):
        return [self.THUMB, self.INDEX, self.MIDDLE, self.RING, self.PINKY]


class Hand:
    WRIST = 0
    ROOT = GoodDict(1, 5, 9, 13, 17)
    PIP = GoodDict(2, 6, 10, 14, 18)
    DIP = GoodDict(3, 7, 11, 15, 19)
    TIP = GoodDict(4, 8, 12, 16, 20)
    FINGER_LIST = ["thumb", "index", "middle", "ring", "pinky"]



class Extract: 
    def __init__(self, graph_features):
        # shape: (frame, feat, dim)
        self.graph_features = graph_features

    def _get_feats(self, feats_list): 
        return self.graph_features[:, feats_list, :]

    @staticmethod
    def _dist_between(a, b, mode="3d"):
        assert a.shape[0] == b.shape[0]
        # 3d = consider all x y z
        distance = np.sqrt(np.sum((a - b)**2, axis=-1))
        return distance

    @staticmethod
    def _point_to_vec(pair): 
        return pair[..., 1, :] - pair[..., 0, :]

    @staticmethod
    def _angle_between(vectors1, vectors2, default_angle=0.0):
        norms1 = np.linalg.norm(vectors1, axis=-1)
        norms2 = np.linalg.norm(vectors2, axis=-1)

        default_mask = np.logical_or(norms1 == 0, norms2 == 0)
        dot_products = np.sum(vectors1 * vectors2, axis=-1)

        cos_angles = dot_products / (norms1 * norms2)
        cos_angles[default_mask] = 1.0

        angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        angles_deg = np.degrees(angles_rad)

        angles_deg[default_mask] = default_angle

        return angles_deg

    def palm(self, z=False): 
        if z: 
            return self.graph_features[:, Hand.WRIST, :]
        else: 
            return self.graph_features[:, Hand.WRIST, :2]
        
    def tip_root_dist(self): 
        return self._dist_between(self._get_feats(Hand.TIP.get_all()), self._get_feats(Hand.ROOT.get_all()))
    
    def root_finger_angle(self): 
        num_fingers = len(Hand.ROOT.get_all())
        # get start and end points
        root_line = self._get_feats([Hand.ROOT.INDEX, Hand.ROOT.PINKY])
        # (frame, 2, 3) -> (frame, finger, 2, 3)
        root_line = root_line[:, np.newaxis, ...]
        # root_line = np.repeat(root_line[:, np.newaxis, ...], num_fingers, 1)

        root_points = self._get_feats(Hand.ROOT.get_all())
        tip_points = self._get_feats(Hand.TIP.get_all())
        finger_lines = np.concatenate(
            (root_points[:, :, np.newaxis, ...], 
             tip_points[:, :, np.newaxis, ...]), 
             axis=2)
        return self._angle_between(
            self._point_to_vec(root_line), 
            self._point_to_vec(finger_lines)
        )
    
    def palm_angle(self): 
        # y-axis upwards normal, perpendicular to the transverse plane
        normal_vector = np.array([0, -1, 0])
        normal_vector = normal_vector[np.newaxis, np.newaxis, :]

        palm_vector = self._point_to_vec(
            self._get_feats([Hand.WRIST, Hand.ROOT.INDEX])
        )[:, np.newaxis, ...]

        return self._angle_between(normal_vector, palm_vector)



class Plotter: 
    @staticmethod
    def plot_line_graph(data, legends, title="Graph Plot", x_axis_label="Frames", y_axis_label="Values", save_path="./test"):
        # Get the number of frames and features from the data shape
        frames, features = data.shape

        # Create a DataFrame for the data with appropriate column names
        df = pd.DataFrame(data, columns=legends)

        # Create the line plot using Plotly Express
        fig = px.line(df, x=np.arange(frames), y=legends, title=title)

        # Customize the plot layout
        fig.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            title={
                'text': title,
                'x': 0.5,  # Align the title to the middle
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial'}
            }
        )

        # with open("{}.png".format(save_path), 'wb') as f:
        #     f.write(to_image(fig, format="png", width=None, height=None, scale=None, validate=True, engine='orca'))

        # Get the HTML code for the plot
        html_code = fig.to_html()

        # fig.write_image("{}.png".format(save_path))

        for labelidx in range(len(legends)): 
            label = legends[labelidx]
            # Plot the arrays
            plt.plot(data[:, labelidx], label=label)

        # Customize the graph
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend()
        plt.savefig('{}.jpg'.format(save_path), format='jpeg', dpi=300)
        plt.close()

        return html_code
    
    @staticmethod
    def plot_spectrogram(data, title="Graph Plot", x_axis_label="Frames", y_axis_label="Feature", save_path="./test"): 
        """
        data is of shape (frame, features, dimensions)
        Since we only draw x and y, only consider the first two of the three dimensions (third dim)
        """
        time_steps, frequencies = data.shape

        # Create the x and y axis values for the spectrogram
        time_axis = np.arange(time_steps)
        freq_axis = np.arange(frequencies)

        # Create the figure and add the heatmap or surface trace
        fig = go.Figure()

        # Using Heatmap (Recommended for large datasets)
        fig.add_trace(go.Heatmap(z=data.T, x=time_axis, y=freq_axis, colorscale='gray'))    # , zmax=1.0, zmin=0.0

        # OR Using Surface (Recommended for small datasets)
        # fig.add_trace(go.Surface(z=data, x=time_axis, y=freq_axis))

        # Customize the plot layout
        fig.update_layout(
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            title={
                'text': title,
                'x': 0.5,  # Align the title to the middle
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial'}
            }
        )

        # with open("{}.png".format(save_path), 'wb') as f:
        #     f.write(to_image(fig, format="png", width=None, height=None, scale=None, validate=True, engine='orca'))

        # Get the HTML code for the plot
        html_code = fig.to_html()

        # fig.write_image("{}.png".format(save_path))

        fig, axs = plt.subplots(1, 1)
        axs.set_title(title)
        axs.set_ylabel(y_axis_label)
        axs.set_xlabel(x_axis_label)
        im = axs.imshow(data.T, origin="lower", aspect="auto", cmap='gray_r')   # , vmin=0.0, vmax=1.0
        fig.colorbar(im, ax=axs)
        plt.savefig('{}.jpg'.format(save_path), format='jpeg', dpi=300)
        plt.close()

        return html_code



    @staticmethod
    def write_to_html(html_code, filename):
        """
        Write the HTML code to an HTML file.

        Parameters:
            html_code (str): The HTML code to be written to the file.
            filename (str): The name of the HTML file to be created.

        Returns:
            None.

        Example usage:
            write_to_html(html_code, 'output_plot.html')
        """
        with open(filename, 'w') as file:
            file.write(html_code)