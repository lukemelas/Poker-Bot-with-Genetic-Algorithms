## A genetic algorithm for finding optimal heuristic AI parameters

from pypokerengine.api.game import setup_config, start_poker
from heuristicAI import HeuristicPlayer
from consoleAI import ConsolePlayer 
import helper
import numpy as np

init_def_prob = np.array([
    [0.6, 0.2, 0.0, 0.2],
    [0.4, 0.4, 0.1, 0.1],
    [0.1, 0.7, 0.2, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.3, 0.7, 0.0]
])

def normalize(narray):
	return [x/sum(x) for x in narray]

class Population(object):
	def __init__(self, size):
		self.pop = []
		self.size = size
		for i in range(size):
			# generate a random bot
			def_prob = normalize(init_def_prob * (1 + np.random.uniform(-0.25, 0.25, size=(5,4))))
			self.pop.append(HeuristicPlayer(def_prob, agg=np.random.uniform(0, 2)))

	def birth_cycle(self):
		""" Conduct a full Moran process, storing the relative fitnesses in a file """
		fitnesses = np.sqrt(self.compute_fitness())
		fitnesses = fitnesses/sum(fitnesses) #normalise to percentages
		new_generation = list(np.random.choice(self.pop, self.size/2, p=fitnesses, replace=False)) #these are the survivors of the Moran process.
		births = np.random.choice(self.pop, self.size - self.size/2, p=fitnesses, replace=True) #these are the new additions
		for new_ai in births:
			if np.random.uniform(0,1) > 0.75:
				new_ai.mutate()
			new_generation.append(new_ai)
		self.pop = new_generation
		print("New generation created!")
		self.print()

	def compute_fitness(self):
		# Divide all the players into 4 tables to play. Play a total of 5 rounds
		total_fitness = [0] * self.size
		for rnd in range(5):
			print("Beginning population round {0}".format(rnd))
			tables = np.random.permutation(self.size)
			table1 = [(self.pop[i], i) for i in tables[:self.size//4]]
			table2 = [(self.pop[i], i) for i in tables[self.size//4:2*self.size//4]]
			table3 = [(self.pop[i], i) for i in tables[2*self.size//4:3*self.size//4]]
			table4 = [(self.pop[i], i) for i in tables[3*self.size//4:]]
			round_fitness = helper.add([self.play_round(table1), self.play_round(table2), self.play_round(table3), self.play_round(table4)])
			print("The fitness totals for this round are: ", round_fitness)
			total_fitness = helper.add([round_fitness, total_fitness])
		return total_fitness

	def play_round(self, players):
		## Input: players is a list of tuples (player, num) where num is the index of player in self.pop
		## Output: a list of their payoffs
		config = setup_config(max_round=20, initial_stack=200, small_blind_amount=1)
		print("Setting up a new table")
		for player, num in players:
			print("Welcoming player {0}".format(num))
			config.register_player(name=num, algorithm=player)
		results = start_poker(config, verbose=0)
		print("The final results of the poker tournament are: ", results)
		fitnesses = [0] * self.size
		for player in results['players']:
			fitnesses[player['name']] = player['stack']
		return fitnesses

	def print(self):
		print([(x.default_prob, x.aggression) for x in self.pop])
		save_file = open("population.txt", "a+")
		save_file.write(str([(x.default_prob, x.aggression) for x in self.pop]))

a = Population(20)
a.print()
for epoch in range(10):
	print("Running epoch {0}".format(epoch))
	a.birth_cycle()
