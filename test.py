"""from game.snake_game import SnakeGame

if __name__ == "__main__":
    agent_ids = [0, 1, 2, 3]
    agent_names = ["DQN", "DQN-ER", "DQN-TN", "DQN-ER-TN"]

    for agent_id in agent_ids:
        agent = load_agent(agent_id, tf.keras.optimizers.Adam(learning_rate=0.001))
        snake_game = SnakeGameWrapper(
            SnakeGame(14, 14, border=1, max_grass=0.05, grass_growth=0.001), 4
        )
        agent.evaluate(snake_game, episodes=100, show_video=False)


def load_agent():
    pass
"""
