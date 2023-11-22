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
Parking Lot Agent Train - Q-learning
'''

from gridworld_env import GridworldEnv
import time
import numpy as np
import sys
import random
import math
from plot_results import plot, plot2

'''
----------------------------------
Parking lot grid map
----------------------------------
0 - black: empty space
1 - gray: barrier
2 - blue: empty parking spot
3 - green: target parking spot
4 - red: car agent start position
----------------------------------
'''
GRID_MAP = [
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 3, 2, 2, 2, 0],
  [0, 1, 1, 1, 1, 0],
  [0, 2, 2, 2, 2, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 4]
]

def main():
    # Initialize the environment
    env = GridworldEnv(grid_map=GRID_MAP, window_title='Homework 2 - Simplified Parking Lot - Training agent')
    env.restart_once_done = True
    env.reward_negative = -0.01
    env.reward_positive = 1

    # Number of discrete states per state dimension
    ENV_SIZE = env.grid_map_shape

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n

    # Learning related constants
    MAX_EXPLORE_RATE = 0.8
    MIN_EXPLORE_RATE = 0.001
    MAX_LEARNING_RATE = 0.99
    MIN_LEARNING_RATE = 0.1
    DECAY_FACTOR = np.prod(ENV_SIZE, dtype=float) / 10.0
    DISCOUNT_FACTOR = 0.99

    # Training constants
    MAX_EPISODES = 5000
    MAX_STEPS_PER_EPISODE = np.prod(ENV_SIZE, dtype=int) * 10
    SOLVINGS_TO_END = 20
    SOLVED_STEPS = np.prod(ENV_SIZE, dtype=int)
    RENDER_ENV = False
    SHOW_PLOT = False
    SHOW_PLOT_END = True
    SHOW_DEBUG_INFO = True
    SHOW_Q_TABLE_END = True
    SIM_SPEED = 0.002
    USE_TRANSFER_KNOWLEDGE = False
    USE_Q_LEARNING = True
    TRAIN_BOTH = False

    # Read command
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    command2 = sys.argv[2] if len(sys.argv) > 2 else ''
    if command == '--agent=ql':
        USE_Q_LEARNING = True
    if command == '--agent=dql':
        USE_Q_LEARNING = False
    if command == '--agent=both':
        TRAIN_BOTH = True
    if command2 == '--render=1':
        RENDER_ENV = True

    # Create Q-learning agent
    agent_ql = QLearning(env=env,
        environment_size=ENV_SIZE, num_actions=NUM_ACTIONS, max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE, show_plot=SHOW_PLOT, show_plot_end=SHOW_PLOT_END,
        max_explore_rate=MAX_EXPLORE_RATE, min_explore_rate=MIN_EXPLORE_RATE,
        max_learning_rate=MAX_LEARNING_RATE, min_learning_rate=MIN_LEARNING_RATE,
        show_q_table_end=SHOW_Q_TABLE_END, solved_steps=SOLVED_STEPS, solvings_to_end=SOLVINGS_TO_END,
        decay_factor=DECAY_FACTOR, render_env=RENDER_ENV, sim_speed=SIM_SPEED, discount_factor=DISCOUNT_FACTOR,
        use_transfer_knowledge=USE_TRANSFER_KNOWLEDGE, show_debug_info=SHOW_DEBUG_INFO)
    
    # Create Double Q-learning agent
    agent_dql = DoubleQLearning(env=env,
        environment_size=ENV_SIZE, num_actions=NUM_ACTIONS, max_episodes=MAX_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE, show_plot=SHOW_PLOT, show_plot_end=SHOW_PLOT_END,
        max_explore_rate=MAX_EXPLORE_RATE, min_explore_rate=MIN_EXPLORE_RATE,
        max_learning_rate=MAX_LEARNING_RATE, min_learning_rate=MIN_LEARNING_RATE,
        show_q_table_end=SHOW_Q_TABLE_END, solved_steps=SOLVED_STEPS, solvings_to_end=SOLVINGS_TO_END,
        decay_factor=DECAY_FACTOR, render_env=RENDER_ENV, sim_speed=SIM_SPEED, discount_factor=DISCOUNT_FACTOR,
        use_transfer_knowledge=USE_TRANSFER_KNOWLEDGE, show_debug_info=SHOW_DEBUG_INFO)

    # Start training
    if TRAIN_BOTH:
        plot_results_ql = agent_ql.start_training()
        plot_results_dql = agent_dql.start_training()
        plot2(plot_results_ql, plot_results_dql)        
    else:
        if USE_Q_LEARNING:
            plot_results_ql = agent_ql.start_training()
            plot(plot_results_ql)
        else:
            plot_results_dql = agent_dql.start_training()
            plot(plot_results_dql, color="red")

    input("Press Enter to continue...")


class QLearning():
    def __init__(self, env, environment_size, num_actions, max_episodes, max_steps_per_episode,
                 show_plot_end, show_debug_info, max_explore_rate, min_explore_rate,
                 max_learning_rate, min_learning_rate, show_plot, show_q_table_end,
                 solvings_to_end, decay_factor, solved_steps, render_env, sim_speed,
                 discount_factor, use_transfer_knowledge):
        self.env = env
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.max_explore_rate = max_explore_rate
        self.min_explore_rate = min_explore_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate = 0
        self.explore_rate = 0
        self.decay_factor = decay_factor
        self.discount_factor = discount_factor
        self.solvings_to_end = solvings_to_end
        self.solved_steps = solved_steps
        self.render_env = render_env
        self.sim_speed = sim_speed
        self.show_plot = show_plot
        self.show_plot_end = show_plot_end
        self.show_q_table_end = show_q_table_end
        self.show_debug_info = show_debug_info
        self.q_table_file = 'trained_q_table.npy'
        if use_transfer_knowledge:
            self.max_explore_rate /= 2
            self.q_table = self.load_q_table()
        else:
            self.q_table = np.zeros(environment_size + (num_actions,), dtype=float)

    def start_training(self):
        # Initiate the learning parameters
        self.learning_rate = self.get_learning_rate(0)
        self.explore_rate = self.get_explore_rate(0)

        num_consecutive_solvings = 0
        plot_rewards = []

        # Render tha maze
        if self.render_env:
            self.env.verbose = True

        for episode in range(self.max_episodes):

            # Reset the environment
            obs = self.env.reset()

            # Set the initial state
            previous_state = tuple(obs)
            total_reward = 0

            for step in range(self.max_steps_per_episode):
                # Select an action: random or with highest q
                action = self.select_action(previous_state)

                # Execute the action
                obs, reward, done, _ = self.env.step(action)

                # Observe the result
                state = tuple(obs)
                total_reward += reward

                # Update the Q based on the result
                best_q = self.update_q_table(previous_state, state, action, reward)

                # Setting up for the next iteration
                previous_state = state

                # Print debug info
                if self.show_debug_info:
                    print("\nEpisode: %d" % (episode+1))
                    print("Steps: %d" % (step+1))
                    print("Action: %d" % action)
                    print("State: %s" % str(state))
                    print("Reward: %f" % reward)
                    print("Best Q: %f" % best_q)
                    print("Explore rate: %f" % self.explore_rate)
                    print("Learning rate: %f" % self.learning_rate)
                    print("Total reward: %f" % total_reward)
                    print("Consecutive solvings: %d" % num_consecutive_solvings)
                    print("")

                # Render tha maze
                if self.render_env:
                    self.env.verbose = True
                    time.sleep(self.sim_speed)

                if done:
                    print("Episode %d finished after %d steps with total reward = %f (consecutive solvings %d)."
                        % (episode+1, step+1, total_reward, num_consecutive_solvings))

                    # Update plot
                    plot_rewards.append(total_reward)
                    if self.show_plot:
                        plot(plot_rewards)

                    if step <= self.solved_steps:
                        num_consecutive_solvings += 1
                    else:
                        num_consecutive_solvings = 0

                    break

                elif step >= self.max_steps_per_episode - 1:
                    print("Episode %d timed out at %d with total reward = %f." % (episode+1, step+1, total_reward))

            # The best policy is considered achieved when solved over SOLVINGS_TO_END times consecutively
            if num_consecutive_solvings > self.solvings_to_end:
                # Save the trained Q table
                self.save_q_table()
                
                # if self.show_plot_end:
                #     plot(plot_rewards)
                #     input("Press Enter to continue...")

                # break
                return plot_rewards

            # Update parameters
            self.explore_rate = self.get_explore_rate(episode)
            self.learning_rate = self.get_learning_rate(episode)


    def update_q_table(self, previous_state, state, action, reward):
        best_q = np.amax(self.q_table[state])
        self.q_table[previous_state + (action,)] += self.learning_rate * (reward + self.discount_factor * (best_q) - self.q_table[previous_state + (action,)])
        return best_q

    def select_action(self, state):
        # Select an action: random or with highest q
        if random.random() < self.explore_rate:
            action = self.env.action_space.sample()
        else:
            action = int(np.argmax(self.q_table[state]))
        return action

    def get_explore_rate(self, t):
        return max(self.min_explore_rate, min(self.max_explore_rate, 1.0 - math.log10((t+1)/self.decay_factor)))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, min(self.max_learning_rate, 1.0 - math.log10((t+1) / self.decay_factor)))

    def save_q_table(self):
        np.save(self.q_table_file, self.q_table)
        if self.show_q_table_end:
            print('\nQ-table')
            print(self.q_table)

    def load_q_table(self):
        return np.load(self.q_table_file)


class DoubleQLearning(QLearning):
    def __init__(self, env, environment_size, num_actions, max_episodes, max_steps_per_episode,
                 show_plot_end, show_debug_info, max_explore_rate, min_explore_rate,
                 max_learning_rate, min_learning_rate, show_plot, show_q_table_end,
                 solvings_to_end, decay_factor, solved_steps, render_env, sim_speed,
                 discount_factor, use_transfer_knowledge):
        super().__init__(env, environment_size, num_actions, max_episodes, max_steps_per_episode,
                 show_plot_end, show_debug_info, max_explore_rate, min_explore_rate,
                 max_learning_rate, min_learning_rate, show_plot, show_q_table_end,
                 solvings_to_end, decay_factor, solved_steps, render_env, sim_speed,
                 discount_factor, use_transfer_knowledge)
        self.q_a_table_file = 'trained_q_a_table.npy'
        self.q_b_table_file = 'trained_q_b_table.npy'
        if use_transfer_knowledge:
            self.max_explore_rate /= 2
            self.q_a_table = self.load_q_a_table()
            self.q_b_table = self.load_q_b_table()
        else:
            self.q_a_table = np.zeros(environment_size + (num_actions,), dtype=float)
            self.q_b_table = np.zeros(environment_size + (num_actions,), dtype=float)

    def update_q_table(self, previous_state, state, action, reward):
        if np.random.rand() < 0.5:
            # Update A
            best_q = np.amax(self.q_b_table[state])
            # a_star = int(np.argmax(self.q_a_table[state]))
            # best_q = self.q_b_table[state + (a_star,)]
            self.q_a_table[previous_state + (action,)] += self.learning_rate * (reward + self.discount_factor * best_q - self.q_a_table[previous_state + (action,)])
        else:
            # Update B
            best_q = np.amax(self.q_a_table[state])
            # b_star = int(np.argmax(self.q_b_table[state]))
            # best_q = self.q_a_table[state + (b_star,)]
            self.q_b_table[previous_state + (action,)] = self.learning_rate * (reward + self.discount_factor * best_q - self.q_b_table[previous_state + (action,)])
        return best_q

    def select_action(self, state):
        # Select an action: random or with highest q
        if random.random() < self.explore_rate:
            action = self.env.action_space.sample()
        else:
            q_table = self.q_a_table[state] + self.q_b_table[state]
            # max_q = np.where(np.max(q_table) == q_table)[0]
            # action = int(np.random.choice(max_q))
            action = int(np.argmax(q_table))
        return action

    def save_q_table(self):
        np.save(self.q_a_table_file, self.q_a_table)
        np.save(self.q_b_table_file, self.q_b_table)
        if self.show_q_table_end:
            print('\nQa-table')
            print(self.q_a_table)
            print('\nQb-table')
            print(self.q_b_table)

    def load_q_a_table(self):
        return np.load(self.q_a_table_file)

    def load_q_b_table(self):
        return np.load(self.q_b_table_file)


if __name__ == "__main__":
    main()