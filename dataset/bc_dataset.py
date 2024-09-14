from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import torch
import pandas as pd
from collections import OrderedDict
import os







class SekiroDataset(Dataset):
    def __init__(self, data_dir, args):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.label_path = [os.path.join(data_dir, f'session_{i}', "label.csv") for i in range(1, 11)]
        self.key2action =  {
            'w': 0,
            'a': 1,
            's': 2,
            'd': 3,
            'k': 4,
            'shift': 5,
            'space': 6,
            'j': 7,
            'r': 8,
            'other': 9    
        }
    
    def transform_state(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        state_transpose = state.transpose((2, 0, 1)) / 255
        state_tensor = torch.tensor(state_transpose, dtype=torch.float32)
        state_tensor = self.transform(state_tensor)
        state_tensor = state_tensor.to(self.device)
        state_tensor = state_tensor.unsqueeze(0)
        return state_tensor 
    
    def read_label(self, label_paths):
        for label_path in label_paths:
            df = pd.read_csv(label_path, header=None)
            keys = df.iloc[0, 2:] 
            for index, row in df.iterrows():
                if index == 0:
                    continue  # Skip the first row
                img_name = row[0]
                for i in range(len(keys)):
                    if float(row[i+2]) == 1:
                        action = self.key2action[keys[i + 2]]
                        self.data.append((os.path.join(self.data_dir, 'images', img_name), action))
                        break
                    else:
                        action = self.key2action["other"]
                        self.data.append((os.path.join(self.data_dir, 'images', img_name), action))
       