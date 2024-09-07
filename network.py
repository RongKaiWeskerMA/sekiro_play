import torch
import torch.nn as nn
import numpy as np



class DQN(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        num_in_feat = resnet.fc.in_features
        resnet.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(num_in_feat, action_space)
                     )
        self.model = resnet
        self.model = self.model.cuda()
        

    def forward(self, cur_state):
        action_prob = self.model(cur_state)

        return action_prob



if __name__ == "__main__":
    dqn = DQN()
    print()