# This is modelling part
# This file is all kinds of tools for modelling only
# LIBS
import datetime

# START
def get_timestamp():
    # for model save
    return datetime.datetime.now().strftime("%m%d%H%M%S")   # timestamp ignores year and second, not really needed. 