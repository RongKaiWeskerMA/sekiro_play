from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
import torch
import pandas as pd
from collections import OrderedDict
import os


class SekiroDataset(Dataset):
    """
    Custom Dataset class for loading Sekiro game frames and corresponding actions.

    Attributes:
        data_dir (str): Directory containing the dataset.
        device (torch.device): Device to run the model on (CPU or CUDA).
        data (list): List of tuples containing frame paths and actions.
        transform (callable): Transformations to be applied on the images.
        label_path (list): List of paths to label CSV files.
        key2action (dict): Mapping of keys to action indices.
    """

    def __init__(self, data_dir, cuda=True):
        """
        Initialize the SekiroDataset with the directory and device.

        Args:
            data_dir (str): Directory containing the dataset.
            cuda (bool): Flag to use CUDA if available.
        """
        self.data_dir = data_dir
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
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
        self.read_label(self.label_path)

    def transform_state(self, state):
        """
        Transform the state image for input to the neural network.

        Args:
            state (numpy.ndarray): Raw state image from the environment.

        Returns:
            torch.Tensor: Preprocessed state tensor.
        """
        state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        state_transpose = state.transpose((2, 0, 1)) / 255
        state_tensor = torch.tensor(state_transpose, dtype=torch.float32)
        state_tensor = self.transform(state_tensor)
        state_tensor = state_tensor.to(self.device)
        return state_tensor 
    
    def read_label(self, label_paths):
        """
        Read the label CSV files and populate the data list with image paths and actions.

        Args:
            label_paths (list): List of paths to label CSV files.
        """
        for session, label_path in enumerate(label_paths, 1):
            df = pd.read_csv(label_path, header=None)
            keys = df.iloc[0, 2:] 
            for index, row in df.iterrows():
                if index == 0:
                    continue  # Skip the first row
                img_name = row[0]
                for i in range(len(keys)):
                    if float(row[i+2]) == 1 and keys[i+2] != 'l':
                        action = self.key2action[keys[i + 2]]
                        self.data.append((os.path.join(self.data_dir, f'session_{session}', 'images', img_name), action))
                        break
                    elif i == len(keys) - 1:
                        action = self.key2action["other"]
                        self.data.append((os.path.join(self.data_dir, f'session_{session}', 'images', img_name), action))

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the image and the corresponding action.
        """
        img_path, action = self.data[idx]
        img = cv2.imread(img_path)
        img = self.transform_state(img)
        label = torch.tensor(action, dtype=torch.long, device=self.device)
        return img, label


if __name__ == "__main__":
    """
    Main entry point for testing the SekiroDataset class.

    This section initializes the dataset and prints the first sample.
    """
    dataset = SekiroDataset(data_dir='data/Sekiro')
    print(dataset[0])   
