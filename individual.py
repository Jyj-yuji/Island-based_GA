import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import Utilities
from bagging import Bagging
from boosting import Boosting
from hyperTuning import Tuner
import numpy as np


class Individual:

    def __init__(self, model, x_train, x_test, y_train, y_test, tuning=False, bagging=True):
        # genes is the list of 0s and 1s to indicate if a feature exists
        self.model = None
        self.fitness_model = model
        self.genes = self.generate_random_genes()
        self.prevent_all_drops()
        self.fitness = 0
        self.accuracy = 0
        self.f1_score = 0
        self.precision_score = 0
        self.recall_score = 0
        self.feature_count = 0
        self.auc_score = 0
        self.tuning = tuning
        self.bagging = bagging
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def generate_random_genes(self):
        genes = [0] * Utilities.GENE_SIZE
        number_to_keep = random.randrange(1, Utilities.GENE_SIZE + 1)
        random_list = random.sample(range(0, Utilities.GENE_SIZE), number_to_keep)
        for num in random_list:
            genes[num] = 1
        return genes

    def get_number_list_from_genes(self):
        my_list = []
        # print(len(self.genes))
        for i in range(0, len(self.genes)):
            # if it is 0, we will drop it, the list contains indexes of all columns to be dropped
            if self.genes[i] == 0:
                my_list.append(i)
        return my_list

    def check_all_drops(self):
        all_zero = True
        for elem in self.genes:
            if elem == 1:
                all_zero = False
        return all_zero

    # We cannot drop all columns, because it will give us nothing to train on!
    # We will randomly change one of the zeros to one in case this happens
    def prevent_all_drops(self):
        if self.check_all_drops():
            index = random.randrange(0, Utilities.GENE_SIZE)
            self.genes[index] = 1

    def get_sum_gain_cost(self):
        mySum = 0.0
        features_to_drop = self.get_number_list_from_genes()
        feature_gain = [Utilities.FEATURE_GAIN[i] for i in range(len(Utilities.FEATURE_GAIN)) if i not in features_to_drop]
        feature_cost = [Utilities.FEATURE_COST[i] for i in range(len(Utilities.FEATURE_COST)) if
                        i not in features_to_drop]
        for i in range(len(feature_gain)):
            mySum += feature_gain[i] * feature_cost[i]
        return mySum

    def calculate_fitness_branch(self):
        # Fitness calculation method: Wa * A + Wf * (P + sum(Ci * Fi))^(-1)
        self.fitness = Utilities.WA * self.accuracy + (1 - Utilities.WA) * (Utilities.P + self.get_sum_gain_cost()) ** (-1)
        # print(self.get_sum_gain_cost())
        return self.fitness

    def update_feature_count(self):
        count = 0
        for i in self.genes:
            if i == 1:
                count += 1
        self.feature_count = count

    def process_bagging(self, clf):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test
        feature_index = self.get_number_list_from_genes()
        x_train = np.delete(x_train, feature_index, axis=1)
        x_test = np.delete(x_test, feature_index, axis=1)
        # TODO: Do HyperParam tuning here
        if self.tuning:
            tuner = Tuner(clf, Utilities.HYPER_PARAMETERS_DT, x_train, y_train, x_test, y_test)
            clf = tuner.best_model
        if self.bagging:
            # do bagging
            ensemble_classifier = Bagging(base_classifier=clf, n_estimators=Utilities.BAGGING_ESTIMATORS)
            ensemble_classifier.fit(x_train, y_train)
        else:
            # do boosting
            ensemble_classifier = Boosting(base_classifier=clf, n_estimators=Utilities.BAGGING_ESTIMATORS)
            ensemble_classifier.fit(x_train, y_train)
        self.f1_score = f1_score(y_test, ensemble_classifier.predict(x_test))
        self.accuracy = accuracy_score(y_test, ensemble_classifier.predict(x_test))
        self.recall_score = recall_score(y_test, ensemble_classifier.predict(x_test))
        self.precision_score = precision_score(y_test, ensemble_classifier.predict(x_test))
        self.auc_score = roc_auc_score(y_test, ensemble_classifier.predict(x_test))
        self.update_feature_count()
        self.calculate_fitness_branch()
        return self.fitness

    def calculate_fitness(self):
        # TODO: move load data function to utilities (V)
        # TODO: Add bagging to the following function (V)
        # TODO: Add HyperParam tuning (V)
        # TODO: Add island-based (V)
        # TODO: Maybe another meta-heuristic (V)
        clf = Utilities.get_model(self.fitness_model)
        # print("accuracy is: " + str(accuracy))
        return self.process_bagging(clf)

    # Crossover: offspring with half genes from one parent and the other half from the second parent
    def crossover(self, partner):
        # print(range(half_size))
        child = Individual(self.fitness_model, self.x_train, self.x_test, self.y_train, self.y_test)
        midpoint = random.randint(0, Utilities.GENE_SIZE - 1)
        child.genes = self.genes[:midpoint] + partner.genes[midpoint:]
        child.prevent_all_drops()
        return child

    # flip a random position
    def mutate(self, mutation_rate):
        for index in range(Utilities.GENE_SIZE):
            if random.uniform(0, 1) < mutation_rate:
                self.genes[index] = (self.genes[index] + 1) % 2
        self.prevent_all_drops()

    def __str__(self):
        return ''.join(str(e) for e in self.genes) + " -> fitness: " + str(self.fitness)
