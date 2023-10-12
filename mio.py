import pickle
import collections
import os
import numpy as np

class AnyIO: 
    # This class is a wrapper to read and write pickle files
    @staticmethod
    def read(IOPath): 
        # only used by loss hists 
        with open(IOPath, 'rb') as f:
            return pickle.load(f)
    @staticmethod
    def save(IOPath, content): 
        with open(IOPath, 'wb') as file:
            pickle.dump(content, file)

class ResultIO: 
    # This class is a wrapper to read and write SolutionOutput objects
    @staticmethod
    def read(IOPath): 
        # only used by loss hists 
        with open(IOPath, 'rb') as f:
            clean_res = pickle.load(f)
        res = collections.namedtuple(
            'SolutionOutputs', clean_res.keys())
        for field in clean_res.keys(): 
            setattr(res, field, clean_res[field])
        return res

    @staticmethod
    def save(IOPath, result): 
        fields = result._fields
        clean_res = {}
        for field in fields: 
            clean_res[field] = getattr(result, field)
        
        with open(IOPath, 'wb') as file:
            pickle.dump(clean_res, file)


class NP_Compress: 
    @staticmethod
    def save(array, filename): 
        np.savez_compressed(filename, data=array)
    
    @staticmethod
    def load(filename): 
        return np.load(filename)["data"]
    

def mk(dir): 
    os.makedirs(dir, exist_ok = True)