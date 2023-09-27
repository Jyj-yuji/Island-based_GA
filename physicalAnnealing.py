import Utilities
from individual import Individual
import random
import math


class PhysicalAnnealing:
    def __init__(self, max_generations, model, x_train, x_test, y_train, y_test, tuning=False):
        self.mutation_rate = Utilities.PA_MUTATION_RATE
        self.max_generations = max_generations
        self.temperature = Utilities.INITIAL_TEMPERATURE
        self.cooling_rate = Utilities.COOLING_RATE
        self.generation = 0
        self.average_fitness = 0.0
        self.max_fitness = 0.0
        self.current_individual = None
        self.best_individual = None
        self.model = model
        self.tuning = tuning
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.visited = []

    def flip_gene(self, gene):
        new_gene = gene.copy()
        for i in range(len(new_gene)):
            if random.random() < self.mutation_rate:
                new_gene[i] = (new_gene[i] + 1) % 2
        return new_gene

    def cool_down(self):
        self.temperature *= self.cooling_rate

    def generate_population(self):
        self.current_individual = Individual(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.tuning)
        self.current_individual.calculate_fitness()
        self.visited.append(self.current_individual.genes)
    
    def start_process(self):
        self.generate_population()
        while self.generation < self.max_generations:
            new_individual = Individual(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.tuning)
            while new_individual.genes in self.visited:
                new_individual.genes = self.flip_gene(self.current_individual.genes)
            new_individual.calculate_fitness()
            if new_individual.fitness > self.current_individual.fitness:
                self.current_individual = new_individual
            else:
                if random.random() < self.acceptance_probability(self.current_individual.fitness, new_individual.fitness):
                    self.current_individual = new_individual
            self.visited.append(self.current_individual.genes)
            self.cool_down()
            self.generation += 1
            if self.current_individual.fitness > self.max_fitness:
                self.max_fitness = self.current_individual.fitness
                self.best_individual = self.current_individual
            self.average_fitness += self.current_individual.fitness
            self.average_fitness /= self.generation
            print("Generation: ", self.generation, " Fitness: ", self.current_individual.fitness, " Temperature: ", self.temperature, "Features: ", self.current_individual.feature_count)
        return self.best_individual

    def acceptance_probability(self, current_fitness, new_fitness):
        return math.exp(((current_fitness - new_fitness) / self.temperature) * (-1))
