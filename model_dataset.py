# This is modelling part

# This is where data is prepared and is loaded into a dataset

# LIBS
from graph_tools import GraphTool
from paths import *
from mio import NP_Compress
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pickle
# START

class DS_Tools:
    @ staticmethod
    def save_indices(filename, my_list):
        try:
            with open(filename, 'wb') as file:
                pickle.dump(my_list, file)
            return True
        except Exception as e:
            print(f"An error occurred while saving the list: {e}")
            return False

    @ staticmethod    
    def read_indices(filename):
        try:
            with open(filename, 'rb') as file:
                my_list = pickle.load(file)
            return my_list
        except Exception as e:
            print(f"An error occurred while reading the list: {e}")
            return None

    @staticmethod
    def cut_frames(arr, cut_range=None):
        """
        Cut a portion of the array along the frame axis between x and y ratios.

        Parameters:
        - arr: NumPy array of shape (frames, 21, 3)
        - cut_start: Start ratio (between 0 and 1)
        - cut_end: End ratio (between 0 and 1)

        Returns:
        - Cut array of shape (new_frames, 21, 3)
        """
        # cut the surroundings if needed, this is for making the testing set
        # it would be best if we have a cleaner data for training, but this might further decrease the data size
        # therefore we only use more largely cut version for testing data 
        if cut_range is None: 
            return arr
        cut_start, cut_end = cut_range
        if cut_start < 0 or cut_end > 1 or cut_start >= cut_end:
            raise ValueError("Invalid cutoff range")

        num_frames = arr.shape[0]
        start_index = int(num_frames * cut_start)
        end_index = int(num_frames * cut_end)

        # Ensure the indices are within bounds
        start_index = max(0, start_index)
        end_index = min(num_frames, end_index)

        return arr[start_index:end_index]

class GuideExtract:
    # this is a messanger class, only carrying some data.
    def __init__(self, monomorph=0, dexter=None, sinister=None):
        self.monomorph = monomorph
        self.dexter = dexter
        self.sinister = sinister
    
    def is_ok(self):
        return self.monomorph == 1 and not (self.dexter and self.sinister)


class GuideReader:
    def __init__(self, guide_path):
        # guide_path should point to a csv file. 
        self.data = pd.read_csv(guide_path)
    
    def __search_by_name__(self, filename):
        return self.data[self.data['NewFileName'] == filename]
    
    @staticmethod
    def __value_acceptable__(value): 
        return value not in ['NONE', 'N/A']
    
    def extract(self, filename): 
        # filename is the video filename
        entry = self.__search_by_name__(filename=filename)

        if entry.empty: 
            return GuideExtract()
        else: 
            # search result not empty
            only_1 = entry['ONLY_1'].iloc[0]
            if only_1 == 1: 
                side = entry['Side'].iloc[0]
                dh_1 = entry['DH_1'].iloc[0]
                oh_1 = entry['OH_1'].iloc[0]

                if side == 'D':
                    right_hand_value = dh_1 if self.__value_acceptable__(dh_1) else None
                    left_hand_value = oh_1 if self.__value_acceptable__(oh_1) else None
                elif side == 'S':
                    left_hand_value = dh_1 if self.__value_acceptable__(dh_1) else None
                    right_hand_value = oh_1 if self.__value_acceptable__(oh_1) else None
                return GuideExtract(
                    monomorph=1, 
                    dexter=right_hand_value, 
                    sinister=left_hand_value
                )
            else: 
                # non-monomorphic
                return GuideExtract()

class HandLandmarkData: 
    def __init__(self, graph_set_dir=None):
        """
        graph_set_dir: the hyper-dir of each video's hand lms (i.e. one dataset)
        """
        if not graph_set_dir: 
            raise Exception("Empty graph set directory! ")
        # init data and tag
        self.data = np.empty((0, 21, 3))
        self.tag = np.array([])

        guidedata = GuideReader(guide_path)
        for clip in os.listdir(graph_set_dir): 
            extract = guidedata.extract(clip)
            if extract.is_ok(): 
                if extract.dexter: 
                    feats = self.__grab_data__(clip, "Right")   # (x, 21, 3)
                    self.data = np.concatenate((self.data, feats))
                    frame, lm, dim = feats.shape
                    tag = np.array([extract.dexter] * frame)
                    self.tag = np.concatenate((self.tag, tag))
                
                if extract.sinister: 
                    feats = self.__grab_data__(clip, "Left")
                    self.data = np.concatenate((self.data, feats))
                    frame, lm, dim = feats.shape
                    tag = np.array([extract.sinister] * frame)
                    self.tag = np.concatenate((self.tag, tag))
            else: 
                # else pass this file, because no data could match
                continue
        print("Data initiated. Count: " + str(self.tag.shape[0]))

    @staticmethod
    def __grab_data__(filename, side): 
        find_name = "{}_{}".format(side, filename)
        gt = GraphTool(graph_dir, find_name)
        gt.delete_empty()

        features = gt.get_features(flatten=False)
        return features
    
    def save_data(self, file_prefix): 
        NP_Compress.save(self.data, os.path.join(data_dir, file_prefix + "_data.npz"))
        NP_Compress.save(self.tag, os.path.join(data_dir, file_prefix + "_tag.npz"))


# Here we define the dataset as will be used in training

class HandshapeDataset(Dataset): 
    def __init__(self, data_path, tag_path):
        self.data = NP_Compress.load(data_path)
        self.tag = NP_Compress.load(tag_path)
        self.dictionary = {tag: index for index, tag in enumerate(sorted(set(self.tag)))}
        
    # REQUIRED: provide size of dataset (= #images)
    def __len__(self) :
        return self.tag.shape[0]

    def __getitem__(self, idx): 
        return self.data[idx], self.dictionary[self.tag[idx]]
    

class HandshapeDict: 
    def __init__(self, tag_path):
        tag = NP_Compress.load(tag_path)
        self.dictionary = {tag: index for index, tag in enumerate(sorted(set(tag)))}
        self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}
    
    def get_dict(self): 
        return self.dictionary
    
    def batch_map(self, class_index_tensor): 
        class_list = [self.reverse_dictionary[index.item()] for index in class_index_tensor]
        return class_list
    
class HandshapeIndexor(Dataset): 
    def __init__(self, tag_path, sign_super_path):
        self.tag = NP_Compress.load(tag_path)
        self.sign_idxes = os.listdir(sign_super_path)
    # REQUIRED: provide size of dataset (= #images)
    def __len__(self) :
        return self.tag.shape[0]

    def __getitem__(self, idx): 
        return self.sign_idxes[idx]




if __name__ == '__main__':
    # training dataset 
    hlmd = HandLandmarkData(graph_set_dir=os.path.join(det_dir, data_name))
    hlmd.save_data(data_name)