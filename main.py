import os
import pygame
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

from agent import Agent, QLearningAgent
from bayesOpt import *
import datetime
import distutils.util

from environment import Environment


#################################
#   Define parameters manually  #
#################################
from screen import Screen


def define_parameters():
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/90
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['episodes'] = 150          
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    # Settings
    params['weights_path'] = 'weights/weights.h5'
    params['load_weights'] = True
    params['train'] = False
    params["test"] = True
    params['plot_score'] = True
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params

def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)    


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = False 
    score, mean, stdev = run(params)
    return score, mean, stdev

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

def run(params, agent:Agent):
    """
    Run the DQN algorithm, based on the parameters previously set.   
    """
    pygame.init()

    env = Environment(440, 440)
    screen = Screen(env)

    counter_games = 0
    score_plot = []
    counter_plot = []
    while counter_games < params['episodes']:
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

        counter_games += 1
        print(f'Game {counter_games}      Score: {env.game.score}')

        score_plot.append(env.game.score)
        counter_plot.append(counter_games)

    if params['plot_score']:
        plot_seaborn(counter_plot, score_plot, params['train'])

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)

    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    params['speed'] = args.speed

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

    run(params, qLearningAgent)
