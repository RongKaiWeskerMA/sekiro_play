import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from itertools import count
import argparse
from network import DQN
import os
import cv2
from env import Sekiro_Env
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import glob
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from dataset.bc_dataset import SekiroDataset
from collections import Counter

class Trainer:
    """
    Main class for training the DQN agent in the Sekiro environment.
    
    Attributes:
        args (argparse.Namespace): Command-line arguments.
        device (torch.device): Device to run the model on (CPU or CUDA).
        env (Sekiro_Env): Environment for training.
        action_space (int): Number of actions in the environment.
        model (DQN): network model.
        optimizer (torch.optim.Optimizer): Optimizer for the network.
        transform (torchvision.transforms.Compose): Image transformations for preprocessing.
        start_epoch (int): Starting epoch number (used for resuming training).
        writer (SummaryWriter): TensorBoard writer for logging.
        best_val_loss (float): Best validation loss observed so far.
    """

    def __init__(self, args):
        """
        Initialize the Trainer with given arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.env = Sekiro_Env()
        print(f"Using device: {self.device}")
        
        self.action_space = self.env.action_space
        self.model = DQN(self.action_space, args.model_type).to(self.device)
        self.optimizer = optim.AdamW([
            {"params": self.model.model.features.parameters(), "lr": args.lr},
            {"params": self.model.model.classifier.parameters(), "lr": args.lr * 10}  
        ], amsgrad=True, weight_decay=0.01)
        self.train_dataset = SekiroDataset(data_dir='data/Sekiro', session_range=18, train_set=True)
        self.train_sampler = self.train_dataset.get_sampler()
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        self.val_dataset = SekiroDataset(data_dir='data/Sekiro', session_range=18, train_set=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        self.class_weights = self.get_class_weights().to(self.device)
        self.start_epoch = 0
        self.best_val_loss = float('inf')  # Initialize best validation loss to infinity
        if args.resume:
            self.load_checkpoint()

        self.writer = SummaryWriter(log_dir='logs')  # Initialize TensorBoard writer with log directory

    def get_class_weights(self):
        """
        Create a weighted sampler to handle imbalanced classes.

        Returns:
            WeightedRandomSampler: A sampler that samples elements according to the specified weights.
        """
        # Count the frequency of each class in the dataset
        class_counts = Counter([label for _, label in self.train_dataset.data])
        num_samples = len(self.train_dataset.data)
        # Calculate the weight for each sample
        class_weights = torch.tensor([num_samples/class_counts[label] for label in range(len(class_counts))])
        # Create a weighted random sampler
        class_weights /= class_weights.sum()

        return class_weights



    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        Args:
        epoch (int): Current epoch number.
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels, weight=self.class_weights)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            self.writer.add_scalar("Training Loss", loss.item(), epoch * len(self.train_loader) + i)
            
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss {loss.item()}")
                
        avg_loss = running_loss / len(self.train_loader)
        avg_accuracy = running_corrects.double() / total_samples
        self.writer.add_scalar("train_accuracy", avg_accuracy, epoch)   
        print(f"End of epoch {epoch}, Average Loss {avg_loss}, Accuracy {avg_accuracy:.4f}")
        
    def validate_one_epoch(self, epoch):
        """
        Validate the model for one epoch.

        Args:
            epoch (int): Current epoch number.
        
        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        val_loss = 0.0
        running_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device) 
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
                
        
                
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_accuracy = running_corrects.double() / total_samples
        print(f"Epoch [{epoch+1}/{self.args.epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
        self.writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        self.writer.add_scalar("val_accuracy", avg_val_accuracy, epoch)
        return avg_val_loss
    
    def train(self):
        """
        Train the model for the specified number of epochs.

        This method handles the training loop, including training and validation
        for each epoch, and saving checkpoints when the validation loss improves.
        """
        for epoch in range(self.start_epoch, self.args.epochs):
            self.train_one_epoch(epoch)
            avg_val_loss = self.validate_one_epoch(epoch)
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch)
                print(f"Checkpoint saved at epoch {epoch} with validation loss {avg_val_loss:.4f}")
        self.writer.close()

    def save_checkpoint(self, epoch):
        """
        Save a checkpoint of the current model state.
        
        Args:
            epoch (int): Current epoch number.
        """
        checkpoint_dir = self.args.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self):
        """
        Load the latest checkpoint to resume training.
        
        This method searches for the most recent checkpoint file and loads
        the model state, optimizer state, and training progress.
        """
        checkpoint_dir = self.args.checkpoint_dir
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
        
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return

        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")

        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.best_val_loss = checkpoint['best_val_loss']
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Optionally, you can load and set the args from the checkpoint
        # self.args = checkpoint['args']
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")

def parse_args():
    """
    Parse command-line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Sekiro RL Training Script")
    
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Number of epochs between checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/bc/", help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--model_type", type=str, default="efficientnet", choices=["efficientnet", "resnet"], help="Type of model to use (efficientnet or resnet)")
    return parser.parse_args()

if __name__ == "__main__":
    """
    Main entry point of the script.
    
    This section parses arguments, sets random seeds for reproducibility,
    initializes the Trainer, and starts the training process.
    """
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    trainer = Trainer(args)
    trainer.train()