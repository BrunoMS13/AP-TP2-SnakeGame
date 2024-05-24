import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from itertools import count

from dqn import DQN
from snake_game import SnakeGame
from replay_memory import ReplayMemory, Transition


# Helper functions for plotting
episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    times_t = torch.tensor(episode_times, dtype=torch.float)

    if show_result:
        plt.title("Result")
        return
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")

    plt.plot(durations_t.numpy(), label="Steps per episode")
    plt.plot(scores_t.numpy(), label="Score per episode", linestyle="dotted")
    plt.plot(times_t.numpy(), label="Time per episode (s)", linestyle="dashed")

    plt.ylabel("Value")
    plt.legend()

    plt.pause(0.001)


# Initialize the environment and the network
input_channels = 3
num_actions = 3
action_mapping = {0: -1, 1: 0, 2: 1}  # Mapping from DQN output to snake game actions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(input_channels, num_actions).to(device)
target_net = DQN(input_channels, num_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4)
memory = ReplayMemory(10000, device)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(num_actions)]], device=device, dtype=torch.long
        )


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


episode_scores = []
episode_times = []
# Training loop
num_episodes = 10
snake_game = SnakeGame(16, 16)

for i_episode in range(num_episodes):
    start_time = time.time()  # Start timing the episode
    state, _, _, info = snake_game.reset()
    state = (
        torch.tensor(state, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    total_score = 0

    for t in count():
        action = select_action(state)
        mapped_action = action_mapping[action.item()]
        observation, reward, terminated, info = snake_game.step(mapped_action)
        reward = torch.tensor([reward], device=device)
        total_score += info["score"]

        if terminated:
            next_state = None
        else:
            next_state = (
                torch.tensor(observation, dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            episode_durations.append(t + 1)
            episode_scores.append(total_score)  # Record the score
            episode_time = time.time() - start_time  # End timing the episode
            episode_times.append(episode_time)  # Record the time in seconds
            plot_durations()
            break

print("Complete")
plot_durations(show_result=True)
plt.ioff()
plt.show()
