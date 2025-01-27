import torch
from torch.utils.data import Dataset
import pandas as pd

class CAV_dataset(Dataset):

    def __init__(self, data, labels):

        assert len(data) == len(labels), "Data and labels must have the same length"

        
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_tensors(self):
        return self.data
    
    def concat(self, other):
        self.data += other.data
        self.labels += other.labels
