import numpy as np


class Bagging:

    def __init__(self, base_classifier, n_estimators=10, max_samples=0.5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.classifiers = []

    def fit(self, x, y):
        n_samples, n_features = x.shape
        n_samples_subset = int(n_samples * self.max_samples)

        for i in range(self.n_estimators):
            # Draw a random subset of the training data
            indices = np.random.choice(n_samples, n_samples_subset, replace=True)
            x_subset, y_subset = x[indices], y[indices]

            # Train the base classifier on the random subset of the data
            classifier = self.base_classifier.fit(x_subset, y_subset)
            self.classifiers.append(classifier)

    def predict(self, x):
        predictions = np.zeros(x.shape[0])
        for classifier in self.classifiers:
            predictions += classifier.predict(x)
        return np.round(predictions / self.n_estimators)
