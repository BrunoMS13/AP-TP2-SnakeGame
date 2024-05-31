import os
import math
import torch
import random
import pickle
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F


class Policy:
    def choose_action(self, state, policy_net, device):
        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
    def __init__(
        self,
        num_actions: int = 3,
        eps_end: float = 0.05,
        eps_decay: int = 25000,
        eps_start: float = 0.95,
    ):
        self.steps_done = 0
        self.num_actions = num_actions

        self.EPS_END = eps_end  # EPS_END is the final value of epsilon
        self.EPS_START = eps_start  # EPS_START is the starting value of epsilon
        self.EPS_DECAY = eps_decay  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay

    def choose_action(self, state, policy_net, device):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.num_actions)]], device=device, dtype=torch.long
            )


# This policy can be useful in situations where you want to balance exploration
# and exploitation and where deterministic policies like epsilon-greedy might not
# provide enough exploration. (Softmax Policy)
class BoltzmannPolicy(Policy):
    def __init__(
        self, initial_temp: float = 1.0, min_temp: float = 0.05, temp_decay: float = 0.999
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.temperature = initial_temp
        self.steps_done = 0

    def choose_action(self, state, policy_net, device):
        with torch.no_grad():
            # Compute action probabilities using softmax
            action_probs = F.softmax(policy_net(state) / self.temperature, dim=1)
            # Sample an action from the probability distribution
            action = torch.multinomial(action_probs, num_samples=1).view(1, 1)

        # Decay the temperature
        self.steps_done += 1
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)

        return action
