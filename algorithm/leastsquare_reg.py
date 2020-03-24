#!/usr/bin/python3

import numpy as np
import pandas as pd

class Least_square():
    """
        Fit a linear model which tries to reduce the residual sum of squares
    between the observed values and the predicted values.
    """

    def __init__(self):
        self.training_size = 0
        self.coef = list()

    def __str__(self):
        return "Multiple_Linear_Regression(Co-effients are: "+str(self.coef)+")"

    def __penalty_term(self):
        term = np.diag(np.ones(self.features+1))
        term[0][0] = 0

        return term

    def __analysis(self, x: np.array, y: np.array) -> list:
        # Does some basic analysis and returns a list of some useful parameters. Format of the list is [self.rse, n-p-1, rsq, adjr2, var_hat]

        yp = x * self.coef

        p = self.features # No. of regressors
        self.rss = sum((y - yp)**2)
        self.rse = np.sqrt(self.rss/(self.training_size - p - 1))
        self.tss = sum((y - np.mean(y))**2)
        self.rsq = 1 - (self.rss/self.tss)
        self.adjr2 = 1 - ((self.rss/(self.training_size - p))/(self.tss/(self.training_size - 1)))
        self.var_hat = self.rss/(self.training_size - 2)

        return

    def get_rsquare(self):
        """
        Returns
        -------
        rsq : float
            Calculated R^2
        """

        return self.rsq

    def get_adjustedr2(self):
        """
        Returns
        -------
        adjr2 : float
            Calculated Adjusted R^2
        """

        return self.adjr2

    def get_params(self):
        """
        Returns
        -------
        coef : ndarray
            returns an 1-d array of coefficients
        """

        return self.coef

    def fit(self, X, y, lambda_value=0.001):
        """
        Parameters
        ----------
        X : ndarray, list
            A n-d array which represents the features.
        y : ndarray, list
            An 1-d array which represents the labels

        Returns
        -------
        coef : ndarray
            returns a n-d array of co-efficients
        """

        self.__lambda = lambda_value
        self.training_size = len(X)
        self.features = X.shape[1]

        self.__penalty_term()

        X = np.append(np.ones((self.training_size, 1)).reshape(-1, 1), np.array(X), axis=1)
        self.coef = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X) + self.__lambda * self.__penalty_term()), X.transpose()), y).T

        return self.coef

    def predict(self, x_new):
        """
        Parameters
        ----------
        x_new : ndarray, list
            An 1-d array which represents the features of an observation

        Returns
        -------
        y_hat : float
            Predicted value of the given observation
        """

        x_new = np.array(x_new)
        y_hat = np.dot(self.coef, x_new)

        return y_hat

if __name__ == "__main__": # This is test code
    data = pd.read_csv('../data/cocomo81.csv')

    y = data[['actual']].values
    data.drop(['actual'], axis=1, inplace=True)
    x = data.values
    reg = Least_square()
    print(reg.fit(x, y))