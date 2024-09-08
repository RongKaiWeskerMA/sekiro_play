import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import argparse
from env import Sekiro_Env
from network import DQN
import os
import cv2
from torchvision import transforms

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.env = Sekiro_Env()
        self.action_space = self.env.action_space
        self.dqn = DQN(self.action_space).to(self.device)
        
        self.policy_net = DQN(self.action_space).to(self.device)
        self.target_net = DQN(self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr, amsgrad=True)
        
        self.memory = ReplayBuffer(10000)  # You might want to make buffer size an argument
        self.steps_done = 0

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_state(self, state):
        # convert opencv state to torch tensor
        state = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        state_transpose = state.transpose((2, 0, 1))
        state_tensor = torch.tensor(state_transpose, dtype=torch.float32)
        state_tensor = self.transform(state_tensor)
        state_tensor = state_tensor.to(self.device)
        state_tensor = state_tensor.unsqueeze(0)
        return state_tensor


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
            np.exp(-1. * self.steps_done / self.args.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.action_space-1)]], device=self.device, dtype=torch.long)

    def train(self):
        for epoch in range(self.args.epochs):
            state = self.env.get_state()
            episode_reward = 0
            state = self.transform_state(state)
            for t in count():
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = self.transform_state(next_state)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                self.optimize_model()
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.args.tau + target_net_state_dict[key] * (1 - self.args.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    self.env.reset()
                    print(f"Episode {epoch+1} finished after {t+1} steps. Total reward: {episode_reward}")   
                    break
                
                if (epoch + 1) % self.args.checkpoint_interval == 0:
                    self.save_checkpoint(epoch + 1)


        print("Training completed.")

    def optimize_model(self):
        if len(self.memory) < self.args.batch_size:
            return
        transitions = self.memory.sample(self.args.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.args.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.policy_optimizer.step()

    def save_checkpoint(self, epoch):
        """Save a checkpoint of the model"""
        checkpoint_dir = self.args.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'args': self.args
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Sekiro RL Training Script")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--eps_start", type=float, default=0.9, help="Starting value of epsilon for epsilon-greedy exploration")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final value of epsilon for epsilon-greedy exploration")
    parser.add_argument("--eps_decay", type=int, default=1000, help="Decay rate for epsilon in epsilon-greedy exploration")
    parser.add_argument("--tau", type=float, default=0.005, help="Update rate for target network")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Number of epochs between checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    trainer = Trainer(args)
    trainer.train()