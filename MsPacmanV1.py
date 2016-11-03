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

prev_reward = 1
max_reward = 0
Trajectories = []
TotalRewards = []

ds = SupervisedDataSet(128, 4)
for i_episode in range(5000):
	observation = env.reset()
	total_reward = 0
	trajectory = []
	for t in range(5000):
		env.render()
		flatObservation = np.array(observation).flatten()
		probs = net.activate(flatObservation)

		observation, reward, done, info = env.step(getAction(probs))
		trajectory.append((flatObservation,np.argmax(probs)))
		total_reward += reward
		if done:
			Trajectories.append(trajectory)
			TotalRewards.append(total_reward)
			
			if (i_episode + 1) % 10 == 0:
				reward_range = max(TotalRewards) - min(TotalRewards)
				threshold = min(TotalRewards) + reward_range*.75
				for i in range(len(Trajectories)):
					if TotalRewards[i] > threshold:
						for state in Trajectories[i]:
							output = []
							for i in range(4):
								if state[1] == i:
									output.append(1)
								else:
									output.append(probs[i])
							ds.addSample(state[0],output)
					else:
						for state in Trajectories[i]:
							output = []
							for i in range(4):
								if state[1] == i:
									output.append(0)
								else:
									output.append(probs[i])
							ds.addSample(state[0],output)
						
				trainer = BackpropTrainer(net,ds)
				trainer.train()
				ds = SupervisedDataSet(128, 4)
			
			print("Episode {} finished with reward {}".format(i_episode + 1, total_reward))
			break
