import gym
import numpy as np
import time
from tensorflow.keras.models import load_model


# Constants
c_env_name = 'Acrobot-v1'

c_trained_model_file_path = './trained_models/trained_v0.h5'

# Globals
g_state_size = None
g_action_size = None

g_env = None
g_model = None


# General utils
def normalize(values, low, high):
    return (values - low) / (high - low)


def get_action(state):
    if np.random.rand() < 0.01:
        return g_env.action_space.sample()
    else:
        q_values = g_model.predict(state)[0]
        return np.argmax(q_values)


def reshape_state(state):
    return np.reshape(state, [1, g_state_size])


def normalize_state(state):
    return normalize(
        state,
        g_env.observation_space.low,
        g_env.observation_space.high)


def preprocess_state(state):
    state = reshape_state(state)
    state = normalize_state(state)
    return state


if __name__ == '__main__':
    g_env = gym.make(c_env_name)

    g_state_size = g_env.observation_space.shape[0]
    g_action_size = g_env.action_space.n

    g_model = load_model(c_trained_model_file_path)

    while True:
        state = reshape_state(g_env.reset())
        done = 0
        step = 0

        while not done:
            g_env.render()
            time.sleep(0.05)

            action = get_action(state)
            next_state, reward, done, info = g_env.step(action)
            next_state = reshape_state(next_state)

            state = next_state
            step += 1

        print('steps: ', step)
        g_env.render()
        time.sleep(2)

