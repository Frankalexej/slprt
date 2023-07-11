from mio import ResultIO
from paths import *
from configs import *
from google.protobuf.json_format import MessageToDict
import json


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
        else: 
            my_dict = {str(i): (0, 0, 0) for i in range(21)} # default (0, 0, 0) for all nodes
        outdict = {"edges": el, "features": my_dict}
        fl.write(json.dumps(outdict, separators=(',', ':')))


if __name__ == '__main__': 
    for vd in os.listdir(det_dir): 
        for clip in os.listdir(det_dir + vd + "/"): 
            vid_graph_dir = graph_dir + vd + "/" + clip + "/"
            # mk(vid_graph_dir)   # ?
            print(clip)
            for g in os.listdir(det_dir + vd + "/" + clip + "/"): 
                clip_dir = det_dir + vd + "/" + clip + "/"
                if g.endswith(".pkl"): 
                    name, ext = g.split(".")
                    d = ResultIO.read(os.path.join(clip_dir, g))
                    side = "Right"
                    json_path = os.path.join(graph_dir, "{}_{}.json".format(side, name))
                    sol2json(d, json_path, side)
                    side = "Left"
                    json_path = os.path.join(graph_dir, "{}_{}.json".format(side, name))
                    sol2json(d, json_path, side)