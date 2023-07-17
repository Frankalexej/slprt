import os
import glob
import json
import numpy as np
from google.protobuf.json_format import MessageToDict
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

        upper_bound = self.features.shape[0]

        # Iterate through each position in the flags array
        for i in range(upper_bound):
            # slide through the array; features and flag are of same length
            if self.flag[i] == FLAG_NONE:   # if this frame is not filled
                start_idx = max(0, i - window_size) # either lower bound of array or edge of window
                end_idx = min(i + window_size + 1, upper_bound) # +1 because end_idx is not considered in slices

                window_flags = self.flag[start_idx:end_idx]
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
                    if FLAG_OK not in self.flag: 
                        # there is nothing to find, then use the old one
                        interpolated_features[i] = self.features[i]
                    else: 
                        # if there is any usable frame to interpolate, find the nearest as the transfer index
                        transfer_index = self._find_closest(self.flag, i, FLAG_OK)
                        interpolated_features[i] = self.features[transfer_index]

                    # # np.min(np.where(window_flags == FLAG_OK)[0])
                    # left_idx = np.argmax(self.flag[:start_idx][::-1] == FLAG_OK)
                    # right_idx = np.argmax(self.flag[end_idx:] == FLAG_OK) + end_idx

                    # if left_idx > -1 and right_idx < self.flag.shape[0]:
                    #     if i - start_idx < right_idx - i:
                    #         nearest_index = start_idx - left_idx - 1
                    #     else:
                    #         nearest_index = right_idx
                    #     interpolated_features[i] = self.features[nearest_index]
                    # elif left_idx > -1:
                    #     nearest_index = start_idx - left_idx - 1
                    #     interpolated_features[i] = self.features[nearest_index]
                    # elif right_idx < self.flag.shape[0]:
                    #     nearest_index = right_idx
                    #     interpolated_features[i] = self.features[nearest_index]
                    # else:
                    #     interpolated_features[i] = interpolated_features[i]

                self.flag[i] = FLAG_FILLED

        return interpolated_features

# Example usage
# Assuming you have the 'flags' and 'features' arrays

# Define the flag values
FLAG_OK = 0
FLAG_NONE = 1

# Specify the window size
window_size = 5

# Perform sliding window interpolation
interpolated_features = sliding_window_interpolation(features, flags, window_size)

# Print the interpolated features
print("Interpolated Features:")
print(interpolated_features)
