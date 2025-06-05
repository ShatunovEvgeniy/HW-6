import cv2
from pathlib import Path
import logging
import torch
import pygame
import os
import datetime

from src.utils.logger import setup_logger
from src.game.wrapped_flappy_bird import GameState, FPS, SCREENWIDTH, SCREENHEIGHT
from src.model.dqn_agent import DQNAgent
from src.utils.device import setup_device

# Set device for computation
device = setup_device()

# Find root path
ROOT_PATH = Path(__file__).parent.parent



def inference(model_path: Path, output_folder: Path):
    """
    Run inference using a trained model and record gameplay video with unique filename.
    :param model_path: Path to the trained model weights.
    :param output_folder: Folder to save the output video.
    """
    # Generate unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"inference_{timestamp}.mp4"
    output_video_path = output_folder / video_filename

    # Setup logger for inference
    logger = setup_logger("Inference", level=logging.DEBUG)
    logger.info(f"Starting inference with model: {model_path}")

    # Create output directory
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize game state
    session = GameState()
    state_size = len(session.get_state())
    action_size = 2

    # Initialize agent and load weights
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()  # Set to evaluation mode
    agent.epsilon = 0  # Disable exploration

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, FPS, (SCREENWIDTH, SCREENHEIGHT))

    # Run inference
    state = session.get_state()
    done = False
    while not done:
        # Get action from model
        action = agent.act(state)
        action_onehot = [1, 0] if action == 0 else [0, 1]

        # Take action
        frame, reward, done = session.frame_step(action_onehot, render=True)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.transpose(frame)  # Rotate to correct orientation
        out.write(frame)

        # Update state
        if not done:
            state = session.get_state()

    # Cleanup
    out.release()
    logger.info(f"Inference complete!")
    logger.info(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    # Setup paths
    model_path = ROOT_PATH / "model" / "lappy_fixed_ep_49800.pth"
    output_folder = ROOT_PATH / "videos"
    os.makedirs(output_folder, exist_ok=True)

    # Run inference
    inference(model_path, output_folder)