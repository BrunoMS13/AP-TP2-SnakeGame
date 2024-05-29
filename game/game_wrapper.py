from collections import deque

import numpy as np

from game.snake_game import SnakeGame


class SnakeGameWrapper:
    def __init__(self, game: SnakeGame, num_frames: int):
        self.game = game
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)

    def reset(self):
        state, reward, done, info = self.game.reset()
        for _ in range(self.num_frames):
            self.frames.append(state)
        return self.get_state(), reward, done, info

    def step(self, action):
        state, reward, done, info = self.game.step(action)
        self.frames.append(state)
        return self.get_state(), reward, done, info

    def get_state(self):
        return self.__preprocess_and_stack_frames(list(self.frames))

    def get_last_state(self):
        return self.frames[-1]

    def __preprocess_and_stack_frames(self, frames: list):
        # do whatever you need to preprocess the frames here
        return np.stack(frames, axis=3)
