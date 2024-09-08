import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model using a pre-trained ResNet18 as the backbone.
    """

    def __init__(self, action_space):
        """
        Initialize the DQN model.

        Args:
            action_space (int): Number of possible actions.
        """
        super().__init__()
        
        # Load pre-trained ResNet18 model
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Get the number of features in the last layer of ResNet
        num_features = efficientnet.classifier[1].in_features
        
        # Replace the final fully connected layer
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, action_space)
        )
        
        self.model = efficientnet

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The action probabilities.
        """
        return self.model(state)


if __name__ == "__main__":
    # Example usage
    action_space = 4  # Example: 4 possible actions
    dqn = DQN(action_space)
    print(f"DQN model created with {action_space} possible actions.")
    
    # You can add more testing or demonstration code here if needed