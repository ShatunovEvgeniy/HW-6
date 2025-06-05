import logging
import os
from pathlib import Path
import torch

from src.game.wrapped_flappy_bird import GameState, PLAYER_HEIGHT, PIPE_HEIGHT, PIPEGAPSIZE
from src.utils.logger import setup_logger
from src.model.dqn_agent import DQNAgent

# Set up logger
logger = setup_logger("Train", level=logging.DEBUG)

# Find root path
ROOT_PATH = Path(__file__).parent.parent
logger.debug(f"Project root: {ROOT_PATH}")

# Set device for computation (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(device)


def train():
    """Training function with fixes for top crash issue"""
    session = GameState()
    initial_state = session.get_state()
    logger.debug(f"Initial state: {initial_state}")

    state_size = len(initial_state)
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    total_episodes = 1000
    max_steps = 10000
    weights_folder = ROOT_PATH / "model"
    os.makedirs(weights_folder, exist_ok=True)

    for episode in range(total_episodes):
        state = session.get_state()
        total_reward = 0
        done = False
        step = 0
        score = 0
        while not done and step < max_steps:
            action = agent.act(state)
            action_onehot = [1, 0] if action == 0 else [0, 1]
            # Take action and get reward
            _, reward, done = session.frame_step(action_onehot, render=False)
            next_state = session.get_state()

            # Add penalty for being too high
            if session.playery < 50:  # Very close to top
                reward -= 0.3

            # Add penalty for not approaching pipes
            pipe_x_diff = session.upperPipes[0]["x"] - session.playerx
            if pipe_x_diff < 50 and abs(session.playery - 200) > 100:
                reward -= 0.2

            # Add reward for proper alignment
            bird_center = session.playery + PLAYER_HEIGHT / 2
            gap_top = session.upperPipes[0]["y"] + PIPE_HEIGHT
            gap_center = gap_top + PIPEGAPSIZE / 2
            if pipe_x_diff < 100 and abs(bird_center - gap_center) < 30:
                reward += 0.3

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            loss = agent.replay()

            # Update state and counters
            state = next_state
            total_reward += reward
            if not done:
                score = session.score
            step += 1

            if done:
                break

        # Print episode summary
        logger.info(f"Episode {episode + 1}: Score={score} Reward={total_reward:.1f} Epsilon={agent.epsilon:.4f}")

        # Save model periodically
        if (episode + 1) % 50 == 0:
            model_path = weights_folder / f"lappy_fixed_ep_{episode + 1}.pth"
            torch.save(agent.model.state_dict(), model_path)

        # Reset the game properly
        session = GameState()


if __name__ == "__main__":
    train()