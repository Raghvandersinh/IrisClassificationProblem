import torch
from torch.utils.data import  Dataset
import pandas as pd

class IrisDataset(Dataset):
    def __init__(self, csv_file):

        self.iris_flowers_dataframe = pd.read_csv(csv_file)
        self.features = self.iris_flowers_dataframe.drop(columns=["Species"]).values
        self.labels = self.iris_flowers_dataframe["Species"].values

    def __len__(self):
        return len(self.iris_flowers_dataframe)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return {"features": torch.tensor(features, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long)}