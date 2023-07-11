from mio import ResultIO
from paths import *
from configs import *
from graph_tools import *

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