import gym
import random
import numpy as np
import pdb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


#OpenAIGym Atari Controls (So we really just need 2, 3, 4, and 5 to make it simple)
# 0 : "NOOP"
# 1 : "FIRE"
# 2 : "UP"
# 3 : "RIGHT"
# 4 : "LEFT"
# 5 : "DOWN"
# 6 : "UPRIGHT"
# 7 : "UPLEFT"
# 8 : "DOWNRIGHT"
# 9 : "DOWNLEFT"
# 10 : "UPFIRE"
# 11 : "RIGHTFIRE"
# 12 : "LEFTFIRE"
# 13 : "DOWNFIRE"
# 14 : "UPRIGHTFIRE"
# 15 : "UPLEFTFIRE"
# 16 : "DOWNRIGHTFIRE"
# 17 : "DOWNLEFTFIRE"

# This converts the index to the Atari action
def getAction(probs):
	index = np.argmax(probs)
	return index + 2

# Sets up the environment. You can change the name to any of the OpenAIGym Environments listed on the website
env = gym.make('CartPole-v0')

# This builds the network
# parameters are the number of neurons in each layer so InputLayer = 4 HiddenLayer = 3 and OutputLayer = 2
# The output layer is set to use a Softmax activation function which is typical for classification networks. (Squashes the output between 0 and 1 so they act as probabilites)
net = buildNetwork(4,3,2, outclass=SoftmaxLayer) 

prev_reward = 1

for i_episode in range(5000): # number of episodes
	observation = env.reset() # need to call this before each episode starts
	total_reward = 0
	trajectory = []


	for t in range(5000): # number of timesteps in each episode

		env.render() # simulates the environment

		probs = net.activate(observation) # this passes the observation from the envrionment through the network

		observation, reward, done, info = env.step(np.argmax(probs)) # Steps to the next state by taken the action with the highest probability

		trajectory.append((observation,probs)) # store the state action pair to make up the dataset later
		total_reward += reward


		if done: # This means the episode has ended
			ds = SupervisedDataSet(4, 2) #sets up the training dataset

			for state in trajectory: # Add samples to the dataset

				output = [0,0]

				if total_reward >= prev_reward: # Good episode increase probability of these actions
					output[np.argmax(state[1])] = 1
					output[np.argmin(state[1])] = state[1][np.argmin(state[1])]

				else: # Bad episode decrease the probabilites of these actions
					output[np.argmax(state[1])] = 0
					output[np.argmin(state[1])] = state[1][np.argmin(state[1])]

				ds.addSample(state[0],output)
			
			trainer = BackpropTrainer(net,ds)
			trainer.train()
			prev_reward = total_reward
			print("Episode finished with reward {}".format(total_reward))
			break
