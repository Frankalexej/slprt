# This is modelling part

# This is where data is prepared and is loaded into a dataset

# LIBS
from graph_tools import GraphTool
from paths import *
import os
# START

class HandLandmarkData: 
    def __init__(self, all_graph_dir):
        """
        all_graph_dir: the hyper-dir of each video's hand lms
        """
        # TODO: work out how to preassume the shape
        self.data = None
        self.tag = None
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

