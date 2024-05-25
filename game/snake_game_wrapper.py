from collections import deque

import numpy as np

from game.snake_game import SnakeGame


class SnakeGameWrapper:
    def __init__(self, game: SnakeGame, k):
        self.game = game
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        state, reward, done, info = self.game.reset()
        for _ in range(self.k):
            self.frames.append(state)
        return self._get_state(), reward, done, info

    def step(self, action):
        state, reward, done, info = self.game.step(action)
        self.frames.append(state)
        return self._get_state(), reward, done, info

    def score(self):
        return self.game.score

    def _get_state(self):
        assert len(self.frames) == self.k
        return self.__preprocess_and_stack_frames(list(self.frames))

    def __preprocess_and_stack_frames(self, frames: list):
        # do whatever you need to preprocess the frames here
        return np.stack(frames, axis=3)
