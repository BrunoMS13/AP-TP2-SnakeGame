from time import sleep
import cv2
from itertools import count

import torch
import torch.nn as nn

from agent.policies import Policy
from game.snake_game import SnakeGame
from agent.replay_memory import ReplayMemory, Transition


class Agent:
    def __init__(
        self,
        device,
        optimizer,
        policy_net,
        policy: Policy,
        snake_game: SnakeGame,
        replay_memory=ReplayMemory(10000),
        target_net=None,
    ):
        self.policy = policy
        self.device = device
        self.optimizer = optimizer
        self.policy_net = policy_net
        self.snake_game = snake_game
        self.target_net = target_net
        self.replay_memory = replay_memory

        self.TAU = 0.005  # TAU is the update rate of the target network
        self.GAMMA = 0.99  # GAMMA is the discount factor
        self.BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer

        self.build_memory(self.snake_game)
        # print(len(self.replay_memory))

    def choose_action(self, state):
        return self.policy.choose_action(state, self.policy_net, self.device)

    def train(self, num_episodes=100, show_video=False):
        if show_video:
            # Create a named window for the display
            cv2.namedWindow("Snake Game Training", cv2.WINDOW_NORMAL)

        scores = 0
        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            pre_state, _, _, info = self.snake_game.reset()

            state = (
                torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            total_score = 0
            for t in count():
                if show_video:
                    cv2.imshow("Snake Game Training", pre_state[:, :, :])
                    cv2.waitKey(1) & 0xFF
                    sleep(0.01)
                action = self.choose_action(state)
                # -1 to assert the action
                pre_state, reward, terminated, info = self.snake_game.step(
                    action.item() - 1
                )
                reward = torch.tensor([reward], device=self.device)
                total_score = info["score"]

                if terminated:
                    next_state = None
                else:
                    next_state = (
                        torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )

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
            if i_episode % 50 == 0:
                print(f"Episode {i_episode} - Avg Score: {scores / 50}")
            """if done:
                episode_durations.append(t + 1)
                plot_durations()
                break"""

        """print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()"""

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
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
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

    def min_distance_heuristic(self, game: SnakeGame) -> int:
        score, apple, head, tail, direction = game.get_state()
        head_x, head_y = head
        apple_x, apple_y = apple[0]
        optimal_direction = 0
        # check what is the "optimal" direction to go to the apple
        if apple_x > head_x:
            optimal_direction = 1
        elif apple_x < head_x:
            optimal_direction = 3
        elif apple_y > head_y:
            optimal_direction = 2
        else:  # apple_y < head_y
            optimal_direction = 0
        # check which way to turn
        if direction == 0:  # up
            if optimal_direction == 1:  # right
                return 2
            elif optimal_direction == 3:  # left
                return 0
        elif direction == 1:  # right
            if optimal_direction == 0:  # up
                return 0
            elif optimal_direction == 2:  # down
                return 2
        elif direction == 2:  # down
            if optimal_direction == 1:  # right
                return 0
            elif optimal_direction == 3:  # left
                return 2
        else:  # direction == 3, left
            if optimal_direction == 0:  # up
                return 2
            elif optimal_direction == 2:  # down
                return 0
        return 1

    def build_memory(self, game: SnakeGame):
        print("Building Memory...")
        game.reset()

        for _ in range(self.replay_memory.memory.maxlen):
            action = self.min_distance_heuristic(game)
            pre_state, reward, done, _ = game.step(action - 1)
            # turn everything into tensors
            state = (
                torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            reward = torch.tensor([reward], device=self.device, dtype=torch.long)
            action = torch.tensor([[action]], device=self.device, dtype=torch.long)

            next_state = None
            if done:
                next_state = None
            else:
                next_state = (
                    torch.tensor(pre_state, dtype=torch.float32, device=self.device)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )

            self.replay_memory.push(state, action, next_state, reward)
            state = next_state

            if done:
                state, _, _, _ = game.reset()
        print("Memory built!")
