import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_dependencies():
    with patch("src.train.GameState") as mock_game, \
            patch("src.train.DQNAgent") as mock_agent, \
            patch("torch.save") as mock_saving, \
            patch("src.train.setup_logger"):
        # Setup mock game
        mock_game.return_value.get_state.return_value = [1.0, 2.0, 3.0, 4.0]
        mock_game.return_value.frame_step.return_value = (None, 1.0, False)
        mock_game.return_value.score = 0

        # Setup mock agent with proper epsilon value
        mock_agent_instance = MagicMock()
        mock_agent_instance.epsilon = 0.5  # Actual float value
        mock_agent_instance.act.return_value = 0
        mock_agent_instance.replay.return_value = 0.5
        mock_agent.return_value = mock_agent_instance

        yield mock_game, mock_agent_instance, mock_saving


def test_training_loop_execution(mock_dependencies):
    from src.train import train

    mock_game, mock_agent, _ = mock_dependencies

    # Run training with minimal episodes/steps
    train(total_episodes=2, max_steps=3, use_wandb=False, render=False)

    # Verify basic execution
    assert mock_game.return_value.get_state.call_count > 0
    assert mock_agent.act.call_count == 6  # 2 episodes * 3 steps
    assert mock_agent.remember.call_count == 6


def test_model_saving(mock_dependencies):
    from src.train import train
    _, mock_agent, torch_save = mock_dependencies

    # Run training that should trigger saving
    train(total_episodes=50, max_steps=1, use_wandb=False, render=False)

    # Verify model saving
    torch_save.assert_called_once()
    assert "lappy_fixed_ep_50.pth" in str(torch_save.call_args[0][1])


def test_early_termination(mock_dependencies):
    from src.train import train
    mock_game, mock_agent, _ = mock_dependencies

    # Make game end on first step
    mock_game.return_value.frame_step.return_value = (None, 0, True)

    # Run training
    train(total_episodes=2, max_steps=100, use_wandb=False, render=False)

    # Verify early termination
    assert mock_agent.act.call_count == 2  # Only 2 actions (1 per episode)
    assert mock_agent.remember.call_count == 2


def test_game_reset(mock_dependencies):
    from src.train import train
    mock_game, _, _ = mock_dependencies

    # Run training
    train(total_episodes=3, max_steps=1, use_wandb=False, render=False)

    # Verify game is reset after each episode
    assert mock_game.call_count == 4  # Initial + 3 resets