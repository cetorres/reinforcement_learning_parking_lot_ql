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
Util plot function to show a chart of 
the learning progress during the agent training
'''

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(rewards, color="blue"):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Homework 2 - Parking lot agent training results')
    # plt.title('Homework 2 - Parking lot agent training results')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(rewards, color=color)
    # plt.ylim(ymin=0)
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.show(block = False)
    plt.pause(.1)

def plot2(rewards1, rewards2):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf()
    manager = plt.get_current_fig_manager()
    manager.set_window_title('Homework 2 - Parking lot agent training results')
    # plt.title('Homework 2 - Parking lot agent training results')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(rewards1, color="blue")
    plt.plot(rewards2, color="red")
    plt.gca().legend(('Q-learning','Double Q-learning'))
    # plt.ylim(ymin=0)
    plt.text(len(rewards1) - 1, rewards1[-1], str(rewards1[-1]))
    plt.text(len(rewards2) - 1, rewards2[-1], str(rewards2[-1]))
    plt.show(block = False)
    plt.pause(.1)