import numpy as np
import pandas as pd

class batchsampler:
    def __init__(self, dataframe, features, labels, batchsize, shuffle = True):
        self.df = dataframe
        self.features = features
        self.labels = labels
        self.ndata = len(dataframe)
        self.shuffle = shuffle
        
        self.batchsize = batchsize
        
        if batchsize == 'all':
            self.batchsize = self.ndata
        
        self.nepoch = -1
        self._i = 0
        self._ii = self.batchsize
        self.newepoch()
        
    def getbatch(self):
        X = self.df[self.features][self._i : self._ii].values
        Y = self.df[self.labels][self._i : self._ii].values
        
        self._i = self._ii
        self._ii = self._ii + self.batchsize
        
        if self._ii > self.ndata:
            self.newepoch()
        
        return X, Y
    
    def newepoch(self):
        if self.shuffle:
            self.df = self.df.reindex(np.random.permutation(self.df.index))
        self._i = 0
        self._ii = self.batchsize
        self.nepoch += 1        
        
        
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, data, features, labels):
        self.n = data.shape[0]
        self.X = data[features].astype(np.float32).values.reshape(-1, len(features))
        self.Y = data[labels].astype(np.float32).values.reshape(-1, len(labels))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]