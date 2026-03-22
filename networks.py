import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ══════════════════════════════════════════════
# ACTOR NETWORK
# ══════════════════════════════════════════════
class Actor(nn.Module):
    """
    Input:  state (4 số thực)
    Output: action [P_common, P_private]
            → dùng softmax để đảm bảo:
              - cả 2 phần ≥ 0
              - tổng = P_total
    """
    def __init__(self, state_dim=4, action_dim=2,
                 hidden_dim=64, P_total=1.0):
        super(Actor, self).__init__()
        self.P_total = P_total

        # 3 lớp fully-connected (giống paper: 4 layers)
        self.fc1 = nn.Linear(state_dim,   hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,  hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,  hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,  action_dim)

    def forward(self, state):
        """
        state: tensor shape (batch_size, 4)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # Softmax: đảm bảo tổng = 1, nhân P_total
        # → tổng công suất = P_total, mỗi phần ≥ 0
        action = F.softmax(x, dim=-1) * self.P_total
        return action


# ══════════════════════════════════════════════
# CRITIC NETWORK
# ══════════════════════════════════════════════
class Critic(nn.Module):
    """
    Input:  (state1, state2, action1, action2)
            = 4 + 4 + 2 + 2 = 12 số
    Output: Q-value (1 số)

    Critic nhìn thấy TOÀN BỘ thông tin:
    → Centralized training
    """
    def __init__(self, state_dim=4, action_dim=2,
                 hidden_dim=64):
        super(Critic, self).__init__()

        # Input = 2 states + 2 actions
        input_dim = state_dim * 2 + action_dim * 2  # = 12

        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)  # output 1 số

    def forward(self, state1, state2, action1, action2):
        """
        Ghép tất cả input lại thành 1 vector
        """
        x = torch.cat([state1, state2,
                        action1, action2], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value