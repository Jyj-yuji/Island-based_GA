import time

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif
from keras.models import Sequential
from keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.adapt import MLkNN
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

FILE_PATH = "NASA_MDP/MDP/D'"

# Genetic Algorithm
POPULATION_SIZE = 30
MUTATION_RATE = 0.03
GENERATION_LIMIT = 50
ELITISM = True
ELITISM_PERCENTILE = 0.2
# Fitness calculation method: Wa * A + Wf * (P + sum(Ci * Fi))^(-1)
# Feature cost is correlation with other columns, feature value is information gain
WA = 0.99 # IF WA == 0.995 -> At most WF provides 0.1 to boost the fitness score over accuracy
P = 0.05
GENE_SIZE = 30
FEATURE_GAIN = []
FEATURE_COST = []
K_NEIGHBORS = 5
BAGGING_ESTIMATORS = 10
# The following are for the island-GA model
ISLAND_COUNT = 3
ISLAND_POPULATION = 30
ISLAND_GENERATION_LIMIT = 30
ISLAND_MIGRATION_RATE = 0.3
ISLAND_MIGRATION_GENERATION = 10
# Base models used in all of genetic algorithms + physical annealing + hill climbing
MODEL_LIST = ["LR", "LDA", "NB", "K-NN", "C4.5", "CART"]
HYPER_PARAMETERS_ON = True
# The following are for the hyper-parameter tuning
SCORING_METRIC = "roc_auc"
HYPER_PARAMETERS_DT = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
}
# The following are for the physical annealing
PA_MUTATION_RATE = 0.1
INITIAL_TEMPERATURE = 1
COOLING_RATE = 0.99
# The following are for the hill climbing
MAX_FEATURE = 5


def get_model(model):
    global K_NEIGHBORS
    if model is None:
        print("no model specified")
        exit()
    if model == "LR":
        return LogisticRegression()
    elif model == "LDA":
        return LinearDiscriminantAnalysis()
    elif model == "NB":
        return GaussianNB()
    elif model == "K-NN":
        return KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    elif model == "K*":
        return MLkNN(n_neighbors=K_NEIGHBORS)
    elif model == "BP":
        # Create a sequential model
        model = Sequential()
        # Add input layer
        model.add(Dense(units=4, input_dim=2, activation='relu'))
        # Add hidden layer
        model.add(Dense(units=4, activation='relu'))
        # Add output layer
        model.add(Dense(units=1, activation='sigmoid'))
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif model == "SVM":
        return svm.SVC(kernel='linear')
    elif model == "C4.5":
        return DecisionTreeClassifier(criterion='entropy')
    elif model == "CART":
        return DecisionTreeClassifier()
    elif model == "RF":
        return RandomForestClassifier()
    else:
        print("such model does not exist")
        exit()


def initialize_global_variables_for_dataset(x, y):
    set_Gene_Size(x)
    set_Feature_Cost(x, y)
    set_Feature_Gain(x, y)


def set_Gene_Size(x):
    global GENE_SIZE
    GENE_SIZE = x.shape[1]


def set_Feature_Cost(x, y):
    global FEATURE_COST
    # Calculate the correlation between each feature and the target variable
    for feature_col in x.transpose():
        # Linear correlation
        correlations, _ = pearsonr(feature_col, y)
        FEATURE_COST.append(1 - abs(correlations))
    # print(FEATURE_COST)


def set_Feature_Gain(x, y):
    global FEATURE_GAIN
    # This is information gain
    FEATURE_GAIN = 1 - mutual_info_classif(x, y)
    # print(FEATURE_GAIN)


def reset_Feature_Cost():
    global FEATURE_COST
    FEATURE_COST = []


def startTimer():
    return time.time()


def endTimer(start):
    end = time.time()
    return end - start
