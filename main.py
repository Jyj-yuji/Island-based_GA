import math
import os
import random

import Utilities
from population import Population
from scipy.io import arff
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from physicalAnnealing import PhysicalAnnealing
from hillClimbing import HillClimbing
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def data_load(file_name):
    data = arff.loadarff("NASA_MDP/MDP/D'/" + file_name)
    df = pd.DataFrame(data[0])
    return x_y_split(df)


def x_y_split(df):
    x = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1].values
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(y_data)
    return x, y


def start_Genetic_Algorithm(filename, model, x_train, x_test, y_train, y_test, tuning=False, bagging=True):
    start = Utilities.startTimer()
    pop = Population(Utilities.MUTATION_RATE, model, x_train, x_test, y_train, y_test, tuning, bagging)
    pop.create_initial_population(Utilities.POPULATION_SIZE)
    my_list = []
    while not pop.finished and pop.generation < Utilities.GENERATION_LIMIT:
        pop.natural_selection()
        pop.generate_new_population()
        pop.evaluate()
        best_fitness = pop.print_population_status()
        my_list.append(best_fitness)
    best_ind = pop.best_individual
    end = Utilities.endTimer(start)
    # print("\n***************************RESULT****************************")
    # print("Time elapsed: ", str(Utilities.endTimer(start)))
    write_to_file(filename, model + "_GA" + "_tuning:" + str(tuning) + "_bagging:" + str(bagging), best_ind.accuracy,
                  best_ind.f1_score, best_ind.precision_score, best_ind.recall_score, best_ind.auc_score, end)


def GA_ISLAND_finished(pop_list):
    for pop in pop_list:
        if not pop.finished:
            return False
    return True


def ga_island_migration(pop_list):
    random_selection = []
    for i in range(Utilities.ISLAND_COUNT):
        # select ISLAND_MIGRATION_RATE of individuals to the next population
        migration_count = math.floor(Utilities.ISLAND_MIGRATION_RATE * Utilities.ISLAND_POPULATION)
        random_selection.append(random.sample(pop_list[i].population, migration_count))
        pop_list[i].population = [j for j in pop_list[i].population if j not in random_selection[i]]
    for i in range(Utilities.ISLAND_COUNT):
        j = (i + 1) % Utilities.ISLAND_COUNT
        pop_list[i].population.extend(random_selection[j])
    return pop_list


def find_best_individual(pop_list):
    best = pop_list[0].best_individual
    for pop in pop_list:
        if pop.best_individual.fitness > best.fitness:
            best = pop.best_individual
    return best


def start_Island_Genetic_Algorithm(filename, model, x_train, x_test, y_train, y_test, tuning=False):
    pop_list = []
    start = Utilities.startTimer()
    for i in range(Utilities.ISLAND_COUNT):
        pop_list.append(Population(Utilities.MUTATION_RATE, model, x_train, x_test, y_train, y_test, tuning))
        pop_list[i].create_initial_population(Utilities.ISLAND_POPULATION)
    while not GA_ISLAND_finished(pop_list) and pop_list[0].generation < Utilities.ISLAND_GENERATION_LIMIT:
        for i in range(Utilities.ISLAND_COUNT):
            pop_list[i].natural_selection()
            pop_list[i].generate_new_population()
            pop_list[i].evaluate()
            best_fitness = pop_list[i].print_population_status()
        # migration of islands every ISLAND_MIGRATION_GENERATION except the first generation
        if pop_list[0].generation % Utilities.ISLAND_MIGRATION_GENERATION == 0 and pop_list[0].generation != 0:
            pop_list = ga_island_migration(pop_list)
    best_ind = find_best_individual(pop_list)
    end = Utilities.endTimer(start)
    # print("\n***************************RESULT****************************")
    # print("Time elapsed: ", str(Utilities.endTimer(start)))
    # Write to file
    write_to_file(filename, model + "_IGA", best_ind.accuracy, best_ind.f1_score, best_ind.precision_score, best_ind.recall_score, best_ind.auc_score, end)
    # print(best_ind.accuracy, best_ind.f1_score, best_ind.precision_score, best_ind.recall_score, best_ind.auc_score)


def start_Regular_Algorithm(filename, model, x_train, x_test, y_train, y_test):
    start = Utilities.startTimer()
    clf = Utilities.get_model(model)
    clf.fit(x_train, y_train)
    # Evaluation
    end = Utilities.endTimer(start)
    accuracy = accuracy_score(y_test, clf.predict(x_test))
    f1 = f1_score(y_test, clf.predict(x_test))
    precision = precision_score(y_test, clf.predict(x_test))
    recall = recall_score(y_test, clf.predict(x_test))
    auc = roc_auc_score(y_test, clf.predict(x_test))
    # Write to file
    write_to_file(filename, model + "_Regular", accuracy, f1, precision, recall, auc, end)
    # print(accuracy, f1, precision, recall, auc)


def write_to_file(file_name, method, accuracy, f1, precision, recall, auc, time):
    with open(file_name, 'a+') as f:
        f.write('****************{}****************\n'.format(method))
        f.write('Accuracy: {}\n'.format(accuracy))
        f.write('F1: {}\n'.format(f1))
        f.write('Precision: {}\n'.format(precision))
        f.write('Recall: {}\n'.format(recall))
        f.write('AUC: {}\n'.format(auc))
        f.write('Time: {}\n'.format(time))


def start_Physical_Annealing(filename, model, x_train, x_test, y_train, y_test, tuning=False):
    start = Utilities.startTimer()
    pa = PhysicalAnnealing(200, model, x_train, x_test, y_train, y_test, tuning)
    best_ind = pa.start_process()
    end = Utilities.endTimer(start)
    write_to_file(filename, model + "_PA", best_ind.accuracy, best_ind.f1_score, best_ind.precision_score, best_ind.recall_score, best_ind.auc_score, end)


def start_Hill_Climbing(filename, model, x_train, x_test, y_train, y_test, tuning=False):
    start = Utilities.startTimer()
    hill = HillClimbing(model, x_train, x_test, y_train, y_test, tuning)
    hill.generate_starting_position()
    while not hill.finished:
        hill.generate_neighbors()
        hill.find_best_neighbor()
        hill.print_hill_climbing_status()
    end = Utilities.endTimer(start)
    # print("time elapsed: " + str(end))
    local_maxima = hill.best_individual
    write_to_file(filename, model + "_HC", local_maxima.accuracy, local_maxima.f1_score, local_maxima.precision_score, local_maxima.recall_score, local_maxima.auc_score, end)


def start_process():
    dir_list = os.listdir(Utilities.FILE_PATH)
    print(dir_list)
    for fileName in dir_list:
        Utilities.reset_Feature_Cost()
        x, y = data_load(fileName)
        print("ones: " + str(np.count_nonzero(y == 1)))
        print("zeros: " + str(len(y) - np.count_nonzero(y)))
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            continue
        Utilities.initialize_global_variables_for_dataset(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        if fileName == "CM1.arff":
        # if fileName == "CM1.arff" or fileName == "KC1.arff" or fileName == "KC3.arff":
        #     model = "CART"
            for model in Utilities.MODEL_LIST:
                start_Regular_Algorithm(fileName + "_result.txt", model, x_train, x_test, y_train, y_test)
                start_Physical_Annealing(fileName + "_result.txt", model, x_train, x_test, y_train, y_test, tuning=False)
                start_Hill_Climbing(fileName + "_result.txt", model, x_train, x_test, y_train, y_test, tuning=False)
                start_Genetic_Algorithm(fileName + "_result.txt", model, x_train, x_test, y_train, y_test, tuning=False,
                                            bagging=True)
                start_Genetic_Algorithm(fileName + "_result.txt", model, x_train, x_test, y_train, y_test, tuning=False,
                                            bagging=False)
                if model == "CART":
                    start_Genetic_Algorithm(fileName + "_result.txt", model, x_train, x_test, y_train, y_test,
                                                tuning=True, bagging=True)
                start_Island_Genetic_Algorithm(fileName + "_result.txt", model, x_train, x_test, y_train, y_test, tuning=False)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    start_process()
