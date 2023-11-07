# This is modelling part

# This is where we put the preset values

# START
dropout = 0.5

in_dim = 63  # 21 x 3
hid_dim = 8 # the number of features we want the model to be learning
out_dim = 93 # number of classes in total

original_dim = 3
length = 21 # number of landmarks

# other latent dimensions in encoder and decoder
enc_lat_dims = [128, 32]
dec_lat_dims = [16, 64]
"""
enc_lat_dims = [32, 16, 8]
dec_lat_dims = [16, 32]
window_sizes = [3, 5, 7, 9, 11]
"""
window_sizes = [3, 5, 7, 9, 11]

# Train configs
BATCH_SIZE = 32
LOADER_WORKER = 16
EPOCHS = 50
BASE = 0