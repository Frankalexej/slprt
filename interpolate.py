import os
import glob

class Interpolator: 
    def __init__(self, data_dir, prefix):
        self.data_dir = data_dir
        self.prefix = prefix

    def _locate_files(self):
        pattern = os.path.join(self.data_dir, self.prefix + '*')
        matching_files = glob.glob(pattern)
        
        # Sort the matching files alphabetically
        matching_files.sort()
        
        return matching_files
    
    def load_data(self): 
        files = self._locate_files()
        for file in files: 
            