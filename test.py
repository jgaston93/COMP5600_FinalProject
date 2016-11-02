import gym
import random
import numpy as np
import pdb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def getAction(probs):
	index = np.argmax(probs)
	return index + 2

env = gym.make('CartPole-v0')

net = buildNetwork(4,3,2, outclass=SoftmaxLayer) 

prev_reward = 1

for i_episode in range(5000):
	observation = env.reset()
	total_reward = 0
	trajectory = []
	for t in range(5000):
		env.render()
		#flatObservation = np.array(observation).flatten()
		probs = net.activate(observation)

		observation, reward, done, info = env.step(np.argmax(probs))
		trajectory.append((observation,probs))
		total_reward += reward
		if done:
			ds = SupervisedDataSet(4, 2)
			for state in trajectory:
				output = [0,0]
				if total_reward >= prev_reward:
					#for i in range(4):
					#	if state[1] == i:
					#		output.append(1)
					#	else:
					#		output.append(0)
					output[np.argmax(state[1])] = 1
					output[np.argmin(state[1])] = state[1][np.argmin(state[1])]
				else:
					output[np.argmax(state[1])] = 0
					output[np.argmin(state[1])] = state[1][np.argmin(state[1])]
					#for i in range(4):
					#	if state[1] == i:
					#		output.append(0)
					#	else:
					#		output.append(1)
				ds.addSample(state[0],output)
			
			trainer = BackpropTrainer(net,ds)
			trainer.train()
			prev_reward = total_reward
			print("Episode finished with reward {}".format(total_reward))
			break
