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

env = gym.make('MsPacman-ram-v0')

net = buildNetwork(128,32,4, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer) 

ds = SupervisedDataSet(128, 4)
for i_episode in range(5000):
	observation = env.reset()
	total_reward = 0
	for t in range(5000):
		env.render()
		flatObservation = np.array(observation).flatten()
		probs = net.activate(flatObservation)

		newobservation, reward, done, info = env.step(getAction(probs))
		total_reward += reward

		output = []
		if reward > 0:
			for i in range(4):
				if np.argmax(probs) == i:
					output.append(1)
				else:
					output.append(probs[i])
		#else:
		#	for i in range(4):
		#		if np.argmax(probs) == i:
		#			output.append(0)
		#		else:
		#			output.append(probs[i])
			
		ds.addSample(observation,output)
		observation = newobservation
		if done:
			
			if (i_episode + 1) % 10 == 0:
						
				trainer = BackpropTrainer(net,ds)
				trainer.train()
				ds = SupervisedDataSet(128, 4)
			
			print("Episode {} finished with reward {}".format(i_episode + 1, total_reward))
			break
