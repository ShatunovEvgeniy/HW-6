import pytest
import numpy as np
import torch
from collections import deque
from unittest.mock import patch
from src.model.dqn_agent import DQN, DQNAgent

# Test configuration overrides
TEST_CONFIG = {
    "deque_maxlen": 100,
    "gamma": 0.99,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "batch_size": 4,  # Smaller for tests
    "update_target_freq": 3
}


@pytest.fixture
def agent():
    """Fixture to create a DQNAgent instance with test configuration"""
    # Patch configuration during tests
    with patch.dict("src.model.hparams.config", TEST_CONFIG):
        state_size = 4
        action_size = 2
        agent = DQNAgent(state_size, action_size)
        # Force CPU for consistent results
        agent.model = agent.model.cpu()
        agent.target_model = agent.target_model.cpu()
        return agent


def test_dqn_forward_pass():
    """Test DQN model architecture and forward pass"""
    model = DQN(input_size=4, output_size=2)
    input_tensor = torch.randn(5, 4)  # Batch of 5 states
    output = model(input_tensor)
    assert output.shape == (5, 2)


def test_agent_initialization(agent):
    """Test agent initialization parameters"""
    assert agent.state_size == 4
    assert agent.action_size == 2
    assert isinstance(agent.memory, deque)
    assert agent.epsilon == 1.0
    assert agent.batch_size == TEST_CONFIG["batch_size"]


def test_act_random(agent):
    """Agent should return random action when epsilon=1"""
    agent.epsilon = 1.0
    state = np.random.randn(4)
    action = agent.act(state)
    assert action in [0, 1]


def test_act_model(agent):
    """Agent should use model for action selection when epsilon=0"""
    agent.epsilon = 0.0
    state = np.array([0.1, -0.2, 0.3, -0.4])
    action = agent.act(state)
    assert action in [0, 1]


def test_remember(agent):
    """Experience should be stored in replay memory"""
    state = [1.0, 2.0, 3.0, 4.0]
    action = 1
    reward = 0.5
    next_state = [1.1, 2.1, 3.1, 4.1]
    done = False

    agent.remember(state, action, reward, next_state, done)
    assert len(agent.memory) == 1
    assert agent.memory[0] == (state, action, reward, next_state, done)


def test_replay_insufficient_memory(agent):
    """Replay should return None when not enough experiences"""
    # Add experiences (less than batch size)
    for _ in range(agent.batch_size - 1):
        agent.remember([0] * 4, 0, 0, [0] * 4, False)

    assert agent.replay() is None


def test_replay_sufficient_memory(agent):
    """Replay should update model and return loss with enough experiences"""
    # Populate memory
    for _ in range(agent.batch_size + 2):
        state = np.random.randn(4).tolist()
        action = np.random.randint(0, 2)
        reward = np.random.random()
        next_state = np.random.randn(4).tolist()
        done = np.random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)

    initial_epsilon = agent.epsilon
    initial_steps = agent.step_count

    loss = agent.replay()

    assert isinstance(loss, float)
    assert agent.epsilon < initial_epsilon
    assert agent.step_count == initial_steps + 1


def test_target_network_update(agent):
    """Target network should update after specified steps"""
    # Set different parameters
    agent.model.fc1.weight.data.fill_(1.0)
    agent.target_model.fc1.weight.data.fill_(0.0)

    # Populate memory and run replay enough times to trigger update
    for _ in range(agent.batch_size):
        agent.remember([0] * 4, 0, 0, [0] * 4, False)

    # Run replay (step_count increases by 1 per call)
    agent.step_count = 2  # Next update at step 3
    agent.replay()  # Now step_count=3

    # Verify parameters are equal after update
    for param, target_param in zip(agent.model.parameters(),
                                   agent.target_model.parameters()):
        assert torch.equal(param.data, target_param.data)


def test_device_consistency(agent):
    """All tensors should be on the correct device"""
    state = np.random.randn(4)
    agent.act(state)  # Trigger tensor creation

    # Verify model parameters are on CPU
    assert next(agent.model.parameters()).device.type == "cpu"
    assert next(agent.target_model.parameters()).device.type == "cpu"

    # Test tensor device in replay
    agent.remember([0] * 4, 0, 0, [0] * 4, False)
    for _ in range(agent.batch_size - 1):
        agent.remember(np.random.randn(4), 0, 0, np.random.randn(4), False)
    agent.replay()

    # Verify optimizer parameters are on CPU
    assert agent.optimizer.param_groups[0]["params"][0].device.type == "cpu"