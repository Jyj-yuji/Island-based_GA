import numpy as np


class Boosting:
    
    def __init__(self, base_classifier, n_estimators):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, x, y):
        n_samples, n_features = x.shape
        weight = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            clf = self.base_classifier
            clf.fit(x, y, sample_weight=weight)
            y_pred = clf.predict(x)
            eps = np.sum(weight[y_pred != y])
            alpha = 0.5 * np.log((1 - eps) / eps)
            self.estimators.append(clf)
            self.alphas.append(alpha)
            # Get weights for next iteration
            weight *= np.exp(-alpha * y * y_pred)
            weight /= np.sum(weight)

    def predict(self, x):
        y_pred = np.zeros(len(x))
        for clf, alpha in zip(self.estimators, self.alphas):
            y_pred += alpha * clf.predict(x)
        return np.sign(y_pred)
