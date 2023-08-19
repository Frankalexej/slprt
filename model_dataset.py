# This is modelling part

# This is where data is prepared and is loaded into a dataset

# LIBS
from graph_tools import GraphTool
from paths import *
import os
import pandas as pd
# START

class GuideExtract:
    # this is a messanger class, only carrying some data.
    def __init__(self, monomorph=0, dexter=None, sinister=None):
        self.monomorph = monomorph
        self.dexter = dexter
        self.sinister = sinister


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
    def __init__(self, all_graph_dir):
        """
        all_graph_dir: the hyper-dir of each video's hand lms
        """
        # TODO: work out how to preassume the shape
        self.data = None
        self.tag = None
        guidedata = GuideReader(guide_path)
        for vd in os.listdir(det_dir): 
            for clip in os.listdir(det_dir + vd + "/"): 
                # here look for the corresponding entry in the annotation file
                # and if 
                for side in ["Right", "Left"]:
                    find_name = "{}_{}".format(side, clip)
                    gt = GraphTool(graph_dir, find_name)
                    # interpolation is done before training, since there is a lot of missing 
                    gt.interpolate(window_size=2) # might not be needed
                    gt.delete_empty()

