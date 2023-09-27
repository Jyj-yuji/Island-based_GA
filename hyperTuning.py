from sklearn.model_selection import GridSearchCV
import Utilities
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class Tuner:
    def __init__(self, model, params, X_train, y_train, X_test, y_test, cv=3, scoring=Utilities.SCORING_METRIC, n_jobs=-1, verbose=1):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.grid_model = GridSearchCV(self.model, param_grid=Utilities.HYPER_PARAMETERS_DT, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs, verbose=self.verbose)
        self.grid_model.fit(self.X_train, self.y_train)
        self.best_params = self.grid_model.best_params_
        self.best_score = self.grid_model.best_score_
        self.best_model = self.grid_model.best_estimator_
        self.y_pred = self.best_model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    def best_params(self):
        return self.best_params

    def best_score(self):
        return self.best_score

    def best_model(self):
        return self.best_model

    def y_pred(self):
        return self.y_pred

    def accuracy(self):
        return self.accuracy