config = dict(
    gamma = 0.99,  # discount factor
    epsilon_min = 0.01,
    epsilon_decay = 0.998,
    learning_rate = 0.0005,
    batch_size = 64,
    update_target_freq = 1000,
    deque_maxlen = 2000,
)