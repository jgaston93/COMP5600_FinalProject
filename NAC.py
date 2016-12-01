from pybrain.rl.learners import ENAC, Reinforce
from pybrain.optimization import HillClimber, FiniteDifferences
from pybrain.rl.agents import LearningAgent, OptimizationAgent
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.experiments import EpisodicExperiment, Experiment
import pdb
from pybrain.rl.explorers.discrete import EpsilonGreedyExplorer
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task
import gym
from pybrain.structure.modules import SoftmaxLayer
from pybrain.rl.environments.cartpole.balancetask import BalanceTask
import numpy
import matplotlib.pyplot as plt

class CartPoleEnv(Environment):
	indim = 1
	outdim = 4

	env = gym.make('CartPole-v0')
	observation = env.reset()
	reward = 0
	totalreward = 0
	all_rewards = []
	done = False
	
	def getSensors(self):
		return self.observation

	def performAction(self, action):
		#self.env.render()
		self.observation, self.reward, self.done, info = self.env.step(action)
		self.totalreward += self.reward
		if self.done:
			print(self.totalreward)
			self.all_rewards.append(self.totalreward)
			self.totalreward = 0
			

	def reset(self):
		self.observation = self.env.reset()
		self.done = False

class CartPoleTask(Task):
	
	def __init__(self, environment):
		self.env = environment

		self.lastreward = 0
	
	def performAction(self, probs):
		action = numpy.argmax(probs)
		self.env.performAction(action)

	def getObservation(self):
		sensors = self.env.getSensors()
		return sensors

	def getReward(self):
		reward = self.env.reward
		cur_reward = self.lastreward
		#pdb.set_trace()
		self.lastreward = reward
		return cur_reward

	def reset(self):
		self.env.reset()

	def isFinished(self):
		#pdb.set_trace()
		return self.env.done

	@property
	def indim(self):
		return self.env.indim

	@property
	def outdim(self):
		return self.env.outdim

env = CartPoleEnv()
task = CartPoleTask(env)

#task = BalanceTask()

net = buildNetwork(task.outdim, 50, task.indim, outclass=SoftmaxLayer)

learner = ENAC()
learner._setLearningRate(.025)
#pdb.set_trace()
agent = LearningAgent(net, Reinforce())

exp = EpisodicExperiment(task,agent)
#pdb.set_trace()
for _ in range(1000):
	#pdb.set_trace()
	
	exp.doEpisodes(1)
	#pdb.set_trace()
	agent.learn()
	agent.reset()

#plt.plot(env.all_rewards)
#plt.show()
