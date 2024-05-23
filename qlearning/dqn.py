import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))

    def predict(self, x, flag):
        with torch.no_grad():
            a = (self.forward(x) * (flag.view(1, -1))).max(1)[1].view(1, 1)
        return a.item()
