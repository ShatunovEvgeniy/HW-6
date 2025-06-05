import logging
import os
from pathlib import Path
import torch
import wandb

from src.game.wrapped_flappy_bird import GameState, PLAYER_HEIGHT, PIPE_HEIGHT, PIPEGAPSIZE
from src.utils.logger import setup_logger
from src.utils.device import setup_device
from src.model.dqn_agent import DQNAgent
from src.model.hparams import config

# Set up logger
logger = setup_logger("Train", level=logging.INFO)

# Find root path
ROOT_PATH = Path(__file__).parent.parent
logger.debug(f"Project root: {ROOT_PATH}")

# Set device for computation
device = setup_device()
logger.info(device)


def train(total_episodes: int = 1000,
          max_steps: int = 1000,
          use_wandb: bool = True,
          render=False
          ) -> None:
    """
    Training process.
    :param total_episodes: Total amount of episodes to train (game count ot play).
    :param max_steps: Maximum steps during episodes.
    :param use_wandb: If True than it logs in wandb.
    :param render: True if visualize train process.
    :return: None.
    """
    session = GameState()
    initial_state = session.get_state()
    logger.debug(f"Initial state: {initial_state}")

    state_size = len(initial_state)
    action_size = 2
    agent = DQNAgent(state_size, action_size)

    weights_folder = ROOT_PATH / "model"
    os.makedirs(weights_folder, exist_ok=True)

    mean_score = 0
    total_score = 0
    for episode in range(total_episodes):
        state = session.get_state()
        total_reward = 0
        done = False
        step = 0
        score = 0
        total_loss = 0
        while not done and step < max_steps:
            action = agent.act(state)
            action_onehot = [1, 0] if action == 0 else [0, 1]
            # Take action and get reward
            _, reward, done = session.frame_step(action_onehot, render=render)
            next_state = session.get_state()

            if mean_score < 0:
                # Add penalty for being too high
                if session.playery < 50:  # Very close to top
                    reward -= 1

                # Add penalty for not approaching pipes
                pipe_x_diff = session.upperPipes[0]["x"] - session.playerx
                if pipe_x_diff < 50 and abs(session.playery - 200) > 100:
                    reward -= 1

                # Add reward for proper alignment
                bird_center = session.playery + PLAYER_HEIGHT / 2
                gap_top = session.upperPipes[0]["y"] + PIPE_HEIGHT
                gap_center = gap_top + PIPEGAPSIZE / 2
                if pipe_x_diff < 100 and abs(bird_center - gap_center) < 30:
                    reward += 1

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            loss = agent.replay()
            if loss is None:
                logger.warning("Loss is None!")
            else:
                total_loss += loss

            # Update state and counters
            state = next_state
            total_reward += reward
            if not done:
                score = session.score
            step += 1

            if done:
                break

        # Print episode summary
        logger.debug(f"Episode {episode + 1}: Score={score} Reward={total_reward:.1f} Epsilon={agent.epsilon:.4f}")
        if use_wandb:
            model_state = dict(
                score = score,
                mean_score = mean_score,
                reward = total_reward,
                epsilon = agent.epsilon,
                loss = total_loss / step,
            )
            wandb.log(model_state, step=episode)

        # Save model periodically
        if (episode + 1) % 50 == 0:
            model_path = weights_folder / f"lappy_fixed_ep_{episode + 1}.pth"
            torch.save(agent.model.state_dict(), model_path)

        # Reset the game properly
        session = GameState()

        total_score += score
        mean_score = total_score / (episode + 1)


if __name__ == "__main__":
    use_wandb = True
    render = False
    if use_wandb:
        wandb.init(project="ml-homework-6",
                   name=f"A lot of parameters")
    train(total_episodes=500000,
          use_wandb=use_wandb,
          render=render)