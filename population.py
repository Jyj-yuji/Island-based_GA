import math
import random


import Utilities
from individual import Individual


class Population:
    def __init__(self, mutation_rate, model, x_train, x_test, y_train, y_test, tuning=False, bagging=True):
        self.model = model
        self.population = []
        self.generation = 0
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.finished = False
        self.perfect_score = 1.0
        self.max_fitness = 0.0
        self.average_fitness = 0.0
        self.mating_pool = []
        self.tuning = tuning
        self.bagging = bagging
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    # Create a random initial population of individuals
    def create_initial_population(self, size):
        for i in range(size):
            ind = Individual(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.tuning, self.bagging)
            ind.prevent_all_drops()
            ind.calculate_fitness()

            if ind.fitness > self.max_fitness:
                self.max_fitness = ind.fitness
                self.best_individual = ind

            self.average_fitness += ind.fitness
            self.population.append(ind)
            self.average_fitness /= size

    # Generate a mating pool based on the individual fitness (probability)
    def natural_selection(self):
        self.mating_pool = []

        for index, ind in enumerate(self.population):
            prob = int(round(ind.fitness * 100))
            self.mating_pool.extend([index for i in range(prob)])

    # Generate a new population from the mating pool
    def generate_new_population(self):
        new_population = []
        pop_size = len(self.population)
        self.average_fitness = 0.0
        # print(pop_size)
        if Utilities.ELITISM:
            number_elitism = int(Utilities.ELITISM_PERCENTILE * len(self.population))
            pop_size -= number_elitism
            new_population += sorted(self.population, key=lambda x: x.fitness, reverse=True)[0:number_elitism]
            # print(number_elitism)

        for i in range(pop_size):
            partner_a, partner_b = self.selection()

            offspring = partner_a.crossover(partner_b)
            offspring.mutate(self.mutation_rate)
            offspring.calculate_fitness()

            self.average_fitness += offspring.fitness
            new_population.append(offspring)

        self.population = new_population
        self.generation += 1
        self.average_fitness /= pop_size

    def selection(self):
        pool_size = len(self.mating_pool)

        i_partner_a = random.randint(0, pool_size - 1)
        i_partner_b = random.randint(0, pool_size - 1)

        i_partner_a = self.mating_pool[i_partner_a]
        i_partner_b = self.mating_pool[i_partner_b]

        return self.population[i_partner_a], self.population[i_partner_b]

    # Evaluate the population
    def evaluate(self):
        best_fitness = 0.0

        for ind in self.population:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                self.max_fitness = best_fitness
                self.best_individual = ind

        if best_fitness == self.perfect_score:
            self.finished = True

    def print_population_status(self):
        print("\nGeneration: " + str(self.generation))
        if Utilities.ELITISM:
            print("----------Elitism is on------------")
        print("Average fitness: " + str(self.average_fitness))
        print("Best individual: " + str(self.best_individual))
        print("Num of features remaining: " + str(self.best_individual.feature_count))
        return self.best_individual.fitness
