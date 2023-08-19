import os
def mk(dir): 
    os.makedirs(dir, exist_ok = True)

# Define here
main_dir = '../'
src_dir = main_dir + "src/"

vid_dir = src_dir + "vid/"

det_dir = src_dir + "det/"

rend_dir = src_dir + "rend/"
rend_vid_dir = rend_dir + "vid/"
rend_pic_dir = rend_dir + "pic/"

graph_dir = src_dir + "graph/"

feats_dir = src_dir + "feats/"

spec_dir = src_dir + "spec/"

spec_pic_dir = src_dir + "spec_pic/"
# End of define

# define file paths
guide_path = src_dir + "HKSL_Citation_Phon_Coding_full_guide.csv"

if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and name.endswith("/") and not name.startswith("__"):
            globals()[name] = mk(value)
            # print(globals()[name])