import torch

from agent.dqn import DQN
from agent.agent import Agent
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
    agent_id: int, show_video: bool = False, agent: Agent | None = None
) -> Agent:
    print(f"Agent {agent_id}")
    game = SnakeGame(14, 14, border=1, max_grass=0.05, grass_growth=0.001)
    snake_game = SnakeGameWrapper(game, num_frames=3)
    snake_game.reset()

    if agent is None:
        policy = EpsilonGreedyPolicy()
        heuristic = MinDistanceHeuristic()
        policy_net = DQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
        target_net = DQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
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
    agent.train(num_episodes=50, show_video=show_video)
    return agent


def main():
    agent_manager = AgentManager()
    agent_id = f"DQN_{3}_StackedFrames_EpsilonGreedy"

    agent = train_agent(0, show_video=True)
    # agent = agent_manager.load(agent_id=agent_id)
    iteration = 0
    # while True:
    print(f"iteration: {iteration}")
    # agent = train_agent(2, show_video=False, agent=agent)

    agent_manager.save(agent, agent_id=agent_id)
    iteration += 1


if __name__ == "__main__":
    main()
