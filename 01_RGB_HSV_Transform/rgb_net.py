import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RGBHSVDataset(Dataset):
    """Custom Dataset for RGB-HSV conversion"""
    def __init__(self, rgb_data, hsv_data):
        self.rgb_data = torch.FloatTensor(rgb_data)
        self.hsv_data = torch.FloatTensor(hsv_data)
        
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.hsv_data[idx]

class RGBtoHSV(nn.Module):
    def __init__(self):
        super(RGBtoHSV, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            # No activation on final layer as HSV has different ranges
        )
    
    def forward(self, x):
        return self.model(x)