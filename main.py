import argparse
import datetime
import distutils.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import seaborn as sns

from agent import Agent, QLearningAgent
from environment import Environment
#################################
#   Define parameters manually  #
#################################
from screen import Screen


def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1 / 90
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200  # neurons in the first layer
    params['second_layer_size'] = 20  # neurons in the second layer
    params['third_layer_size'] = 50  # neurons in the third layer
    params['episodes'] = 150
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['load_weights'] = True
    params['train'] = False
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + '.txt'
    return params


def plot_metrics(metrics, filepath=None):
    formatted_dict = {'episodes': [],
                      'metrics': [],
                      'results': []}

    n = len(metrics['episodes'])
    for i in range(n):
        episode = metrics['episodes'][i]
        score = metrics['scores'][i]
        reward = metrics['rewards'][i]

        formatted_dict['episodes'].append(episode)
        formatted_dict['metrics'].append('score')
        formatted_dict['results'].append(score)

        formatted_dict['episodes'].append(episode)
        formatted_dict['metrics'].append('reward')
        formatted_dict['results'].append(reward)

    df_metrics = pd.DataFrame(formatted_dict)
    sns.lineplot(data=df_metrics, x='episodes', y='results', hue='metrics')
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)


def decode_state(encoded_state):
    """
    encoded_state: an array of 0s and 1s representing a binary value

    return: decimal value
    """
    decoded = ''
    for s in encoded_state:
        decoded += str(s)

    return int(decoded, 2)


def decode_action(encoded_action):
    if isinstance(encoded_action, np.ndarray):
        return encoded_action.argmax()
    return encoded_action


def run(params, agent: Agent):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()

    env = Environment(440, 440)
    screen = Screen(env)

    episode = 0
    metrics = {'episodes': [],
               'scores': [],
               'rewards': []}

    while episode < params['episodes']:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if params['display']:
            screen.display()

        state1, done = env.reset()
        state1 = decode_state(state1)
        action1 = agent.choose_action(state1)
        episode_reward = 0
        while not done:
            # Getting the next state, reward
            state2, reward, done = env.step(action1)
            state2 = decode_state(state2)
            # Choosing the next action
            action2 = agent.choose_action(state2)

            # Learning the Q-value
            decoded_action1 = decode_action(action1)
            decoded_action2 = decode_action(action2)
            agent.update(state1, state2, reward, decoded_action1, decoded_action2)

            state1 = state2
            action1 = action2
            episode_reward += reward

            if params['display']:
                screen.display()
                pygame.time.wait(params['speed'])

        episode += 1
        print(f'Game {episode}      Score: {env.game.score}')

        mean_reward = episode_reward/params['episodes']
        metrics['episodes'].append(episode)
        metrics['rewards'].append(mean_reward)
        metrics['scores'].append(env.game.score)

    return metrics

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--episodes", nargs='?', type=int, default=150)
    parser.add_argument("--figure", nargs='?', type=str, default=None)

    args = parser.parse_args()
    print("Args", args)

    params = dict()
    params['display'] = args.display
    params['speed'] = args.speed
    params['episodes'] = args.episodes

    # Defining all the required parameters
    epsilon = 0.1
    total_episodes = 500
    max_steps = 100
    alpha = 0.5
    gamma = 1

    action_space = np.eye(3)
    num_actions = 3
    num_state = 2 ** 11
    qLearningAgent = QLearningAgent(epsilon, alpha, gamma, num_state, num_actions, action_space)

    metrics = run(params, qLearningAgent)
    plot_metrics(metrics, filepath=args.figure)


