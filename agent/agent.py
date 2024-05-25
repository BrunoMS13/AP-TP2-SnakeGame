from time import sleep
import time
import cv2
from itertools import count

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from agent.policies import Policy
from game.snake_game_wrapper import SnakeGameWrapper
from agent.replay_memory import ReplayMemory, Transition
from agent.heuristics import min_distance_heuristic, do_not_hit_yourself_heuristic

class Agent:
    def __init__(
        self,
        device,
        optimizer,
        policy_net,
        policy: Policy,
        game_wrapper: SnakeGameWrapper,
        target_net=None,
    ):
        self.policy = policy
        self.device = device
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.game_wrapper = game_wrapper
        self.target_net = target_net
        self.replay_memory = ReplayMemory(10000, device)

        self.replay_memory.build_memory(game_wrapper, min_distance_heuristic)

        self.TAU = 0.005  # TAU is the update rate of the target network
        self.GAMMA = 0.99  # GAMMA is the discount factor
        self.BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer

    def choose_action(self, state):
        return self.policy.choose_action(state, self.policy_net, self.device)

    def train(self, num_episodes=100, show_video=False):
        if show_video:
            # Create a named window for the display
            cv2.namedWindow("Snake Game Training", cv2.WINDOW_NORMAL)

        scores = 0

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            pre_state, _, _, info = self.game_wrapper.reset()

            state = torch.tensor(pre_state, dtype=torch.float32, device=self.device)

            total_score = 0
            for t in count():
                if show_video:
                    cv2.imshow("Snake Game Training", pre_state[:, :, :])
                    cv2.waitKey(1) & 0xFF
                    sleep(0.01)
                action = self.choose_action(state)
                # -1 to assert the action
                pre_state, reward, terminated, info = self.game_wrapper.step(
                    action.item() - 1
                )
                reward = torch.tensor([reward], device=self.device)
                total_score = info["score"]
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(pre_state, dtype=torch.float32, device=self.device)

                # Store the transition in memory
                self.replay_memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    break
            scores += total_score
            if (i_episode+1) % 50 == 0:
                print(f"Episode {i_episode+1} - Avg Score: {scores / 50}")
                scores = 0

        

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        # Pad tensors with extra dimensions, every state should have the same shape, which is [batch_size, width, height, channels, num_frames]
        padded_next_states = []
        for next_state in batch.next_state:
            if next_state is not None:
                padded_next_states.append(next_state)
            else:
                padded_next_states.append(torch.zeros_like(batch.state[0]))
        # Concatenate padded tensors
        non_final_next_states = torch.cat(padded_next_states, dim=0)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
