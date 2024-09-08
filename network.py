import torch
import torch.nn as nn

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
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Get the number of features in the last layer of ResNet
        num_features = resnet.fc.in_features
        
        # Replace the final fully connected layer
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, action_space)
        )
        
        self.model = resnet

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