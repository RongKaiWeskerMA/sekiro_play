import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model using a pre-trained EfficientNet or ResNet as the backbone.
    """

    def __init__(self, action_space, model_type='efficientnet'):
        """
        Initialize the DQN model.

        Args:
            action_space (int): Number of possible actions.
            model_type (str): Type of model to use ('efficientnet' or 'resnet').
        """
        super().__init__()

        if model_type == 'efficientnet':
            # Load pre-trained EfficientNet-B0 model
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, action_space)
            )
        elif model_type == 'resnet':
            # Load pre-trained ResNet18 model
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, action_space)
            )
        else:
            raise ValueError("Invalid model type. Choose 'efficientnet' or 'resnet'.")

        self.model = model

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
    dqn = DQN(action_space, model_type='efficientnet')
    print(f"DQN model created with {action_space} possible actions using EfficientNet.")
    
    dqn_resnet = DQN(action_space, model_type='resnet')
    print(f"DQN model created with {action_space} possible actions using ResNet.")
    
    # You can add more testing or demonstration code here if needed