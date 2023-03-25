from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class My_Dataset(Dataset):
    def __init__(self, filename):
        super(My_Dataset).__init__()
        self.filePath = f'Hand Landmarks data/{filename}'
        self.data = pd.read_csv(self.filePath)
        self.X_train = self.data.iloc[0:, 1: -1]
        self.y_train = self.data.iloc[0:, -1]
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        return self.X_train.iloc[index, :].tolist(), self.y_train.iloc[index].tolist()

    def __len__(self):
        return self.len
