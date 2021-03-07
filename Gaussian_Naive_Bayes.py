# File Name: Gaussian_Naive_Bayes.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")


class NaiveBayes:
    """ Gaussian naïve Bayes """
    """
        Independent feature model, that is, the naïve Bayes probability model:
             Bayesian probability
                --> posterior = likelihood . prior / evidence
                --> p(y|X) = p(X|y) . p(y) / p(X)
        In practice, there is interest only in the numerator of that fraction,
        because the denominator does not depend on y and the values of the
        features xi are given, so that the denominator is effectively constant.
        
        With feature vector  X = x1, x2, ..., xn and all features are mutually independent then:
            p(y|X) = p(x1|y).p(x2|y). ... .p(xn|y) . p(y) / p(X)
        """

    def fit(self, x_trn, y_trn):
        # What is the sample size and number of features?
        n_samples, n_features = x_trn.shape
        # How many classes in the training data set?
        self._classes = np.unique(y_trn)
        # What is the length of the classes?
        n_classes = len(self._classes)

        # Initialize mean, var, and prior
        self.mu = np.zeros((n_classes, n_features), dtype=np.float64)
        self.sigma = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Segment the data by the class, and then compute the mean, variance, prior
        # of x in each class
        for index, cls in enumerate(self._classes):
            # Take the all x values if y = cls
            x_values = x_trn[y_trn == cls]
            # Calculate mean, variance for each feature of x_values
            self.mu[index, :] = x_values.mean(axis=0)
            self.sigma[index, :] = x_values.var(axis=0)
            # Calculate prior for a given class - - frequency
            #   --> p(y) = number of samples in the class/total number of samples
            self._priors[index] = x_values.shape[0] / float(n_samples)

    def predict(self, x_test):
        # Predict the labels for each feature in x_test
        y_predicted = [self.max_posterior(x) for x in x_test]

        # Return the predicted labels
        return np.array(y_predicted)

    def max_posterior(self, xs):
        """ Constructing a classifier from the probability model """
        """
            likelihood is the y-axis value for a fixed data point 
            with a distribution that can be moved: l(moving distribution | fixed data point) 
                    
            posterior =  p(y|X) =  likelihood . p(y) 
                      =  l(x1|y).l(x2|y). ... .l(xn|y) . p(y)
                    
            But, to avoid the underflow problem ( a computer limit of a very 
            small numbers when they get closer to 0), we apply the log function:
                  
            posterior =  log p(y|X)= log likelihood + log p(y)
                      =  log l(x1|y) + log l(x2|y) + ... + log l(xn|y) + log p(y)
            
            The naïve Bayes classifier combines this model with a decision rule.
            One common rule is to pick the hypothesis that is most probable;
            this is known as the maximum a posteriori or MAP decision rule.
            
            max posterior = argmax_y( log p(y|X))= argmax_y(log likelihood + log p(y))
                   =  argmax_y(log l(x1|y) + log l(x2|y) + ... + log l(xn|y) + log p(y))
        """
        posteriors = []

        # Calculate posterior for each class
        for index, cls in enumerate(self._classes):
            # Calculate the log of prior
            log_prior = np.log(self._priors[index])
            # Calculate the sum of log likelihood for each value in xs (xs is one dimensional list)
            log_likelihood = np.sum(np.log(self.pdf(index, xs)))
            posterior = log_likelihood + log_prior
            posteriors.append(posterior)

        # Return the class with maximum posterior probability
        return self._classes[np.argmax(posteriors)]

    def pdf(self, class_index, xi):
        """ Probability Density Function """
        """
        When dealing with continuous data, a typical assumption is that the
        continuous values associated with each class are distributed according
        to a normal (or Gaussian) distribution. We first segment the data by the class,
        and then compute the mean and variance of x in each class.
        Suppose we have collected some observation value v.
        Then, the probability density of v given a class y, p(x=v|y), can be computed by plugging
        v into the equation for a normal distribution parameterized by mean and variance . That is:
           p(v = xi|y) = pdf = 1 / sqrt(2 . π . σ^2) . e^(-((xi - μ)^2 / 2 . σ^2))
        """

        # Get the mean and Var of xi (xi is a single value)
        mean = self.mu[class_index]
        var = self.sigma[class_index]
        # Calculate the probability density for xi (xi is a single value)
        numerator = np.exp(- np.power(xi - mean, 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        _pdf = numerator / denominator

        # return the pdf
        return _pdf


def accuracy(y_tst, y_predicted):
    acc = np.sum(y_tst == y_predicted) / len(y_tst)
    return acc


if __name__ == '__main__':
    # Make a classification using sklearn
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=10,
                                        n_classes=2,
                                        random_state=123)
    # Split X and y to training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=123)
    # Create a
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test,
                                                          y_predictions))
