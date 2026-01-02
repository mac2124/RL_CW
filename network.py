import torch
import torch.nn as nn

class CNNPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.actor(x), self.critic(x)