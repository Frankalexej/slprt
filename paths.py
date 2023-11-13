import os
def mk(dir): 
    os.makedirs(dir, exist_ok = True)

# Define here
main_dir = '../'
src_dir = main_dir + "src/"
code_dir = main_dir + "codes/"

vid_dir = src_dir + "vid/"

det_dir = src_dir + "det/"

rend_dir = src_dir + "rend/"
rend_vid_dir = rend_dir + "vid/"
rend_pic_dir = rend_dir + "pic/"

graph_dir = src_dir + "graph/"

feats_dir = src_dir + "feats/"

spec_dir = src_dir + "spec/"

spec_pic_dir = src_dir + "spec_pic/"

data_dir = src_dir + "data/"

preds_dir = src_dir + "preds/"

static_resources_dir = src_dir + "static_resources/"
handshape_rendering_dir = static_resources_dir + "hs/"

model_save_dir = main_dir + "model_save/"

__relative_static_resources_dir = "../../../static_resources/"
# End of define

# define file paths
guide_path = src_dir + "HKSL_Citation_Phon_Coding_full_guide.csv"
predict_result_path = src_dir + "HKSL_Prediction.xlsx"

hidview_style_path = __relative_static_resources_dir + "hidview.css"
predsview_style_path = __relative_static_resources_dir + "predsview.css"
predsview_js_path = __relative_static_resources_dir + "predsview.js"


# End of define

# define names
data_name = "Cynthia_full"
train_name = "Cynthia_train"
test_mono_name = "Cynthia_test_mono"
test_poly_name = "Cynthia_test_poly"

data_train_name = data_name + "_train"
data_validation_name = data_name + "_val"

if __name__ == '__main__':
    # For all paths defined, run mk()
    for name, value in globals().copy().items():
        if isinstance(value, str) and value.endswith("/") and not name.startswith("__"):
            globals()[name] = mk(value)
            # print(globals()[name])