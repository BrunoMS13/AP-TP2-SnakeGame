import os
import pickle
from copy import deepcopy

from agent.agent import Agent


class AgentManager:
    def __init__(self, agent_dir="saved_agents"):
        self.agent_dir = agent_dir

    def save(self, agent: Agent, agent_id: str):
        print(f"Saving agent {agent_id}")
        # Create agent dir
        agent_dir = f"{self.agent_dir}{os.sep}{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)
        agent_copy = deepcopy(agent)
        with open(f"{agent_dir}{os.sep}agent.pickle", "wb") as file:
            pickle.dump(agent_copy, file)

    def load(self, agent_id: str) -> Agent:
        print(f"Loading agent {agent_id}")
        agent_dir = f"{self.agent_dir}{os.sep}{agent_id}"
        # Get agent
        with open(f"{agent_dir}{os.sep}agent.pickle", "rb") as file:
            agent: Agent = pickle.load(file)

        return agent
