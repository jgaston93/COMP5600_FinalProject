import gym
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer,SigmoidLayer

def getAction(probs):
	return np.argmax(probs)

alpha = 10
num_episodes = 3000
training_epochs = 10000
env = gym.make('CartPole-v0')

net = buildNetwork(4,3,2, hiddenclass=SigmoidLayer,outclass=SoftmaxLayer) 

for j in range(training_epochs):
	gradient = []
	for i in range(len(net.params)):
		total_reward = 0
		reward1 = 0
		reward2 = 0
		
		observation = env.reset()
		for t in range(num_episodes):
			probs = net.activate(observation)
			observation, reward, done, info = env.step(getAction(probs))
			total_reward += reward
			if done:
				break
				
		reward1 = total_reward
		total_reward = 0
		
		net.params[i] += alpha
		
		observation = env.reset()
		
		for t in range(num_episodes):
			probs = net.activate(observation)
			observation, reward, done, info = env.step(getAction(probs))
			total_reward += reward
			if done:
				break
		
		reward2 = total_reward
		if reward2 > reward1:
			gradient.append(alpha*.5)
		elif reward1 > reward2:
			gradient.append(-alpha*.5)
		else:
			gradient.append(0)
		net.params[i] -= alpha

	for i in range(len(net.params)):
		net.params[i] += gradient[i]

	observation = env.reset()
	total_reward = 0
	for t in range(num_episodes):
		env.render()
		probs = net.activate(observation)
		observation, reward, done, info = env.step(getAction(probs))
		total_reward += reward
		if done:
			break
	print("Episode finished with reward {}".format(total_reward))
		
