import Utilities
import random
from individual import Individual


class HillClimbing:
    def __init__(self, model, x_train, x_test, y_train, y_test, tuning=False):
        self.generation = 0
        self.best_individual = None
        self.finished = False
        self.perfect_score = 1.0
        self.max_fitness = 0.0
        self.neighbors = []
        self.model = model
        self.tuning = tuning
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def regen_genes(self):
        genes = [0] * Utilities.GENE_SIZE
        number_to_keep = random.randrange(1, min(Utilities.GENE_SIZE, Utilities.MAX_FEATURE) + 1)
        random_list = random.sample(range(0, Utilities.GENE_SIZE), number_to_keep)
        for num in random_list:
            genes[num] = 1
        return genes

    def generate_starting_position(self):
        ind = Individual(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.tuning)
        ind.genes = self.regen_genes().copy()
        ind.calculate_fitness()
        self.best_individual = ind
        self.max_fitness = ind.fitness
        if self.max_fitness == self.perfect_score:
            self.finished = True

    def find_best_neighbor(self):
        no_better_neighbor = True
        for ind in self.neighbors:
            if ind.fitness > self.max_fitness:
                no_better_neighbor = False
                self.max_fitness = ind.fitness
                self.best_individual = ind
                if self.max_fitness == self.perfect_score:
                    self.finished = True
        if no_better_neighbor:
            print("There is no better neighbors, ending!")
            self.finished = True
        else:
            self.generation += 1

    def generate_neighbors(self):
        current_list = self.best_individual.genes
        # print(current_list)
        # print("original ^")
        for i in range(Utilities.GENE_SIZE):
            ind = Individual(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.tuning)
            ind.genes = current_list.copy()
            ind.genes[i] = (ind.genes[i] + 1) % 2
            if not ind.check_all_drops():
                # print(ind.genes)
                ind.calculate_fitness()
                self.neighbors.append(ind)

    def print_hill_climbing_status(self):
        print("\nGeneration: " + str(self.generation))
        print("Max fitness: " + str(self.max_fitness))
        print("Best individual: " + str(self.best_individual))
        return self.best_individual