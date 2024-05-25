import torch

from agent.dqn import DQN
from agent.agent import Agent
from agent.agent_manager import AgentManager
from agent.policies import EpsilonGreedyPolicy
from game.snake_game_wrapper import SnakeGameWrapper, SnakeGame


LR = 1e-3  # Optimizer learning rate
NUM_ACTIONS = 3  # turn left, right or continue
INPUT_CHANNELS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_agent(agent_id: int, show_video=False):
    print(f"Agent {agent_id}")
    snake_game = SnakeGame(30, 30, border=1, max_grass=0.05, grass_growth=0.001)
    snake_game_wrapper = SnakeGameWrapper(snake_game, k=1)
    snake_game_wrapper.reset()

    policy = EpsilonGreedyPolicy()
    policy_net = DQN(INPUT_CHANNELS*1, NUM_ACTIONS).to(device)
    target_net = DQN(INPUT_CHANNELS*1, NUM_ACTIONS).to(device)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR)

    agent = Agent(
        policy=policy,
        device=device,
        optimizer=optimizer,
        policy_net=policy_net,
        target_net=target_net,
        game_wrapper=snake_game_wrapper,
    )
    agent.train(num_episodes=1000, show_video=show_video)
    return agent


def main():
    ag_id = 1
    agent_manager = AgentManager()
    agent = train_agent(2, show_video=False)
    agent_manager.save(agent, agent_id=f"DQN_{ag_id}")

    """ Example of saving and loading an agent """
    # old_agent = agent_manager.load(agent_id=f"DQN_{ag_id}")
    # old_agent.train(num_episodes=20, show_video=False)
    # agent_manager.save(old_agent, agent_id=f"DQN_{ag_id}")


if __name__ == "__main__":
    main()
