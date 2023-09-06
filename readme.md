# SLPRT: Sign Language Praat Project

## How to run through the codes
1. run paths.py
2. put the videos under src/vid/xxx/
3. run lm_detect.py
4. run graph_extract.py
5. run extract_plot.py / run model_dataset.py


## How to train the model
1. run paths.py
2. put the videos under src/vid/xxx/
3. run lm_detect.py
4. run graph_extract.py
5. run random_split.py: randomly split the data (defaultly named as "Cynthia_full") into 0.8:0.2 train:validation sets. 
6. run model_dataset.py