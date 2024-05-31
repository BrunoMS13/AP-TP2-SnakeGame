import cv2
import torch
from time import sleep
from itertools import count

from agent.agent import Agent
from agent.dqn import DQN, ReducedDQN, BigDQN, DuelingDQN
from game.snake_game import SnakeGame
from agent.agent_manager import AgentManager
from game.game_wrapper import SnakeGameWrapper
from agent.heuristics import MinDistanceHeuristic
from agent.policies import EpsilonGreedyPolicy, BoltzmannPolicy


LR = 1e-4  # Optimizer learning rate
NUM_ACTIONS = 3  # turn left, right or continue
INPUT_CHANNELS = 3 * 3  # 3 stacked frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_agent(
    agent_id: int, show_video: bool = False, agent: Agent | None = None, num_episodes=1000
) -> Agent:
    print(f"Agent {agent_id}")
    game = SnakeGame(14, 14, border=1, max_grass=0.05, grass_growth=0.001)
    snake_game = SnakeGameWrapper(game, num_frames=3)
    snake_game.reset()

    if agent is None:
        policy = BoltzmannPolicy()
        heuristic = MinDistanceHeuristic()
        policy_net = ReducedDQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
        target_net = ReducedDQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
        optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR)

        agent = Agent(
            policy=policy,
            device=device,
            optimizer=optimizer,
            heuristic=heuristic,
            policy_net=policy_net,
            snake_game=snake_game,
            target_net=target_net,
        )
    agent.snake_game = snake_game
    agent.train(num_episodes=num_episodes, show_video=show_video)
    return agent


def test_agent(agent: Agent, episodes=10, show_video=True):
    scores = []
    if show_video:
        cv2.namedWindow("Snake Game", cv2.WINDOW_NORMAL)
    max_score = 0
    for _ in range(episodes):
        pre_state, reward, done, _ = agent.snake_game.reset()
        for _ in count():
            if show_video:
                cv2.imshow("Snake Game", pre_state[:, :, :, -1])
                cv2.waitKey(1) & 0xFF
                sleep(0.01)
            state = (
                torch.tensor(pre_state, dtype=torch.float32, device=device)
                .permute(2, 3, 0, 1)
                .reshape(-1, agent.snake_game.game.width+2*agent.snake_game.game.border, agent.snake_game.game.height+2*agent.snake_game.game.border)
                .unsqueeze(0)
            )
            action = agent.choose_action(state)
            # -1 to assert the action
            pre_state, reward, terminated, info = agent.snake_game.step(
                action.item() - 1
            )
            if terminated:
                scores.append(info["score"])
                if info["score"] > max_score:
                    max_score = info["score"]
                break
    print(f"Avg score: {sum(scores) / episodes}")
    print(f"Highest score: {max_score}")


def main():
    agent_manager = AgentManager()
    agent_id = f"Test15"

    agent = train_agent(agent_id=agent_id, show_video=False, num_episodes=1000)
    # agent: Agent = agent_manager.load(agent_id=agent_id)
    # while True:
    #     agent = train_agent(agent_id=agent_id, show_video=False, agent=agent, num_episodes=100)
    #     agent_manager.save(agent, agent_id=agent_id)
    # train_agent(agent_id=agent_id, show_video=False, agent=agent, num_episodes=1000)
    # agent_manager.save(agent, agent_id=agent_id)

    agent.plot_scores()


    # agent: Agent = agent_manager.load(agent_id=agent_id)
    # agent.snake_game.game = SnakeGame(14, 14, border=1)
    # test_agent(agent, episodes=100, show_video=False)


if __name__ == "__main__":
    main()
