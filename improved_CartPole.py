import gym
import random
import numpy as np
import pdb
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer,SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def getAction(probs):
	index = np.argmax(probs)
	return index + 2

env = gym.make('CartPole-v0')

net = buildNetwork(4,3,2, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer) 

prev_reward = 1
max_reward = 0
Episodes = []

ds = SupervisedDataSet(4, 2)
gooddataset = SupervisedDataSet(4,2)
goodexamples = 0
for i_episode in range(5000):
	observation = env.reset()
	total_reward = 0
	trajectory = []
	for t in range(5000):
		env.render()
		#flatObservation = np.array(observation).flatten()
		probs = net.activate(observation)

		newobservation, reward, done, info = env.step(np.argmax(probs))
		trajectory.append((observation,probs))
		observation = newobservation
		total_reward += reward
		if done:
			for state in trajectory:
				output = [0,0]
				if total_reward >= 100:
					output[np.argmax(state[1])] = 1
					output[np.argmin(state[1])] = 0
					gooddataset.addSample(state[0],output)
					goodexamples += 1
				else:
					output[np.argmax(state[1])] = 0
					output[np.argmin(state[1])] = 1
				ds.addSample(state[0],output)
			if (i_episode + 1) % 100 == 0:
				trainer = BackpropTrainer(net,ds)
				trainer.train()
				if goodexamples > 0:
					trainer = BackpropTrainer(net,gooddataset)
					trainer.trainEpochs(3)
				ds = SupervisedDataSet(4, 2)
			
			print("Episode {} finished with reward {}".format((i_episode + 1), total_reward))
			break
