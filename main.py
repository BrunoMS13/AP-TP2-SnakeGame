import torch

from agent.dqn import DQN
from agent.agent import Agent
from game.snake_game import SnakeGame
from agent.agent_manager import AgentManager
from agent.policies import EpsilonGreedyPolicy


LR = 1e-4  # Optimizer learning rate
NUM_ACTIONS = 3  # turn left, right or continue
INPUT_CHANNELS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_agent(agent_id: int):
    print(f"Agent {agent_id}")
    snake_game = SnakeGame(14, 14, border=1, max_grass=0.05, grass_growth=0.001)
    snake_game.reset()

    policy = EpsilonGreedyPolicy()
    policy_net = DQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
    target_net = DQN(INPUT_CHANNELS, NUM_ACTIONS).to(device)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR)

    agent = Agent(
        policy=policy,
        device=device,
        optimizer=optimizer,
        policy_net=policy_net,
        snake_game=snake_game,
        target_net=target_net,
    )
    agent.train(num_episodes=1000)
    return agent


def main():
    agent = train_agent(0)

    """ Example of saving and loading an agent """
    agent_manager = AgentManager()
    agent_manager.save(agent, agent_id=f"DQN_{0}")
    # old_agent = agent_manager.load(agent_id=f"DQN_{0}")
    # old_agent.train(num_episodes=50)


if __name__ == "__main__":
    main()
