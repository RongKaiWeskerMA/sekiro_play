from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import cv2
import torch
import pandas as pd
from collections import OrderedDict, Counter
import os
from PIL import Image

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

    def __init__(self, data_dir, session_range, train_set=True, cuda=True):
        """
        Initialize the SekiroDataset with the directory and device.

        Args:
            data_dir (str): Directory containing the dataset.
            session_range (int): Range of sessions to include in the dataset.
            train_set (bool): Flag to indicate if the dataset is for training or validation.
            cuda (bool): Flag to use CUDA if available.
        """
        self.data_dir = data_dir
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if train_set:
            self.label_path = [os.path.join(data_dir, f"session_{i}", "label.csv") for i in range(1, session_range+1)]
        else:
            self.label_path = [os.path.join(data_dir, f"session_{i}", "label.csv") for i in range(session_range+1, 21)]
        
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
        
        state_tensor = self.transform(state)
       
        return state_tensor 
    
    def read_label(self, label_paths):
        """
        Read the label CSV files and populate the data list with image paths and actions.

        Args:
            label_paths (list): List of paths to label CSV files.
        """
        for label_path in label_paths:
            session = label_path.split("\\")[1]
            df = pd.read_csv(label_path, header=None)
            keys = df.iloc[0, 2:] 
            for index, row in df.iterrows():
                if index == 0:
                    continue  # Skip the first row
                img_name = row[0]
                for i in range(len(keys)):
                    if float(row[i+2]) == 1 and keys[i+2] != 'l':
                        action = self.key2action[keys[i + 2]]
                        self.data.append((os.path.join(self.data_dir, session, 'images', img_name), action))
                        break
                    elif i == len(keys) - 1:
                        action = self.key2action["other"]
                        self.data.append((os.path.join(self.data_dir, session, 'images', img_name), action))

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
        img = Image.open(img_path)
        img = self.transform_state(img)
        label = torch.tensor(action, dtype=torch.long)
        return img, label

    def get_sampler(self):
        """
        Create a weighted sampler to handle imbalanced classes.

        Returns:
            WeightedRandomSampler: A sampler that samples elements according to the specified weights.
        """
        # Count the frequency of each class in the dataset
        class_counts = Counter([label for _, label in self.data])
        num_samples = len(self.data)
        
        # Calculate the weight for each sample
        weights = [1.0 / class_counts[label] for _, label in self.data]
        
        # Create a weighted random sampler
        sampler = WeightedRandomSampler(weights, num_samples)
        return sampler


if __name__ == "__main__":
    """
    Main entry point for testing the SekiroDataset class.

    This section initializes the dataset and prints the first sample.
    """
    dataset = SekiroDataset(data_dir='data/Sekiro', session_range=18, train_set=True)
    sampler = dataset.get_sampler()
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
    for data in dataloader:
        print(data)
        break
