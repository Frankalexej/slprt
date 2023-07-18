import os
import glob
import json
import numpy as np
import pandas as pd
from google.protobuf.json_format import MessageToDict
import plotly.express as px
from scipy.spatial import distance as d
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
    def interpolate(self, window_size=2):
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


class Hand:
    WRIST = 0
    ROOT = [1, 5, 9, 13, 17]
    PIP = [2, 6, 10, 14, 18]
    DIP = [3, 7, 11, 15, 19]
    TIP = [4, 8, 12, 16, 20]
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
    
    def palm(self, z=False): 
        if z: 
            return self.graph_features[:, 0, :]
        else: 
            self.graph_features[:, 0, :2]
        
    def tip_root_dist(self): 
        return self._dist_between(self._get_feats(Hand.TIP), self._get_feats(Hand.ROOT))
    
    def 


class Plotter: 
    @staticmethod
    def plot_line_graph(data, legends, title="Graph Plot", x_axis_label="Frames", y_axis_label="Values"):
        """
        Plot a line graph of a sequence of data using Plotly Express.

        Parameters:
            data (numpy.ndarray): The data to be plotted, with shape (frames, features).
            legend (list): A list containing the legend names for each feature.
            x_axis_label (str): Label for the x-axis. Default is "Frames".
            y_axis_label (str): Label for the y-axis. Default is "Values".
            title (str): Title of the plot. Default is "Line Graph".

        Returns:
            None (displays the plot).

        Example usage:
            data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            legend = ['Feature 1', 'Feature 2', 'Feature 3']
            plot_line_graph(data, legend)
        """
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

        # Get the HTML code for the plot
        html_code = fig.to_html()

        # Show the plot
        # fig.show()
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