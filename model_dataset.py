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
import torch
# START

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
    def __init__(self, all_graph_dir=det_dir):
        """
        all_graph_dir: the hyper-dir of each video's hand lms
        """
        # init data and tag
        self.data = np.empty((0, 21, 3))
        self.tag = np.array([])

        guidedata = GuideReader(guide_path)
        for vd in os.listdir(all_graph_dir): 
            for clip in os.listdir(all_graph_dir + vd + "/"): 
                extract = guidedata.extract(clip)
                if extract.is_ok(): 
                    if extract.dexter: 
                        feats = self.__grab_data__(clip, "Right")   # (x, 21, 3)
                        self.data = np.concatenate((self.data, feats))
                        frame, lm, dim = feats.shape
                        tag = np.array([extract.dexter] * frame)
                        self.tag = np.concatenate((self.tag, tag))
                    
                    if extract.sinister: 
                        self.__grab_data__(clip, "Left")
                        self.data = np.concatenate((self.data, feats))
                        frame, lm, dim = feats.shape
                        tag = np.array([extract.dexter] * frame)
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
        return gt.get_features(flatten=False)
    
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




if __name__ == '__main__':
    hlmd = HandLandmarkData()
    hlmd.save_data("cynthia_data")