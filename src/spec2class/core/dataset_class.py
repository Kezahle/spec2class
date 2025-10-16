import torch
import pandas as pd
import numpy as np 

class Dataset(torch.utils.data.Dataset):
      #Characterizes a dataset for PyTorch

    def __init__(self, list_DBs, x_dict):
        'Initialization'
        self.list_DBs = list_DBs
        self.x_dict = x_dict

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_DBs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        try:
            DB = self.list_DBs[index]
            x = self.x_dict[index]
            return x, DB
        except KeyError:
            pass


