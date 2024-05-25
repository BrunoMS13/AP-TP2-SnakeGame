import random
from collections import namedtuple, deque
from typing import Callable

import torch
from game.snake_game_wrapper import SnakeGameWrapper, SnakeGame

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.device = device
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def build_memory(self, wrapper: SnakeGameWrapper, heuristic_function: Callable[[SnakeGame], int]):
        print("Building Memory...")
        state, _, _, _ = wrapper.reset()
        for _ in range(self.memory.maxlen):
            action = heuristic_function(wrapper.game)
            state_next, reward, done, _ = wrapper.step(action - 1)
            # turn everything into tensors
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_tensor = torch.tensor([[action]], device=self.device)
            state_next_tensor = torch.tensor(state_next, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward_tensor = torch.tensor([reward], device=self.device)
            self.push(state_tensor, action_tensor, state_next_tensor, reward_tensor)
            state = state_next
            if done:
                state, reward, done, _ = wrapper.reset()
        print("Memory built!")
