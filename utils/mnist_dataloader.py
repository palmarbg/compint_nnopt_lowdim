import pandas as pd
import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, file, transform=None):
        df = pd.read_csv(file)

        self.label = torch.tensor(df[df.columns[0]].to_numpy(), dtype=torch.long).reshape((-1,))
        self.data = torch.tensor(df[df.columns[1:]].to_numpy() / 255, dtype=torch.float32).reshape((-1, 784))

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        X = self.data[idx, :]
        Y = self.label[idx]

        if self.transform:
            X = self.transform(X.reshape(1,28,28)).reshape(784)
            
        return X, Y