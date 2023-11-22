'''
University of Colorado at Colorado Springs
PhD in Computer Science

Class: CS 4080-5080 - Reinforcement Learning - Fall 2021
Professor: Jugal Kalita
Student: Carlos Eugenio Lopes Pires Xavier Torres
Student ID: 110320128
E-mail: clopespi@uccs.edu
Date: October 27, 2021

Homework 2
Agent Test
'''

import gym
from gridworld_env import GridworldEnv
import numpy as np
import math
import time
from agent_train import GRID_MAP


def start_testing():

    # Render tha maze
    env.verbose = True

    for episode in range(MAX_EPISODES):

        # Reset the environment
        obs = env.reset()

        # input('Enter to continue')

        # Set the initial state
        previous_state = tuple(obs)
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):

            # Select an action
            action = select_action(previous_state)

            # Execute the action
            obs, reward, done, _ = env.step(action)

            # Observe the result
            state = tuple(obs)
            total_reward += reward
           
            # Setting up for the next iteration
            previous_state = state

            # Render tha maze
            time.sleep(SIM_SPEED)             

            if done:
                print("Episode %d finished after %d steps with total reward = %f."
                      % (episode+1, step+1, total_reward))
                break

            elif step >= MAX_STEPS_PER_EPISODE - 1:
                print("Episode %d timed out at %d with total reward = %f." % (episode+1, step+1, total_reward))



def select_action(state):
    action = int(np.argmax(q_table[state]))
    return action


def load_q_table(q_table_file):
    return np.load(q_table_file)


if __name__ == "__main__":
    # Initialize the maze environment
    env = GridworldEnv(grid_map=GRID_MAP, window_title='Homework 2 - Simplified Parking Lot - Testing agent')
    env.restart_once_done = True

    # Testing constants
    MAX_EPISODES = 5
    MAX_STEPS_PER_EPISODE = 100
    SIM_SPEED = 0.2

    # Load trained Q table
    q_table = load_q_table('trained_q_table_P1.npy')
    
    # Start agent testing
    start_testing()