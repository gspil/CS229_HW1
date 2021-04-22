import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        X_transpose = np.transpose(X)
        inverse_X_transpose = np.linalg.inv(np.matmul(X_transpose, X))
        X_Transpose_Y = np.matmul(X_transpose, y)
        self.theta = np.matmul(inverse_X_transpose, X_Transpose_Y)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = X.shape[0];

        map = np.zeros((n_examples, k + 1))

        for i in range (0, n_examples):
            # skip intercept value to get x
            x = X[i][1]

            for j in range (0, k + 1):
                map[i][j] = np.power(x, j)

        return map
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        n_examples = X.shape[0];

        map = np.zeros(n_examples, k + 2)

        for i in range (0, n_examples):
            # skip intercept value to get x
            x = X[i][1]

            for j in range (0, k + 1):
                map[i][j] = np.power(x, j)

            map[i][k + 1] = np.sin(x)

        return map
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples = X.shape[0];

        predictions = np.zeros(n_examples)

        for i in range (0, n_examples):

            predictions[i] = np.matmul(np.transpose(self.theta), X[i])

        return predictions
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)



    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel()

        poly_train = lm.create_poly(k, train_x)
        poly_predict = lm.create_poly(k, plot_x)

        lm.fit(poly_train, train_y)

        plot_y = lm.predict(poly_predict)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp(train_path, ks=[3],filename='plot_4b.png')

    run_exp(train_path, ks=[3, 5, 10, 20],filename='plot_4c.png')

    run_exp(train_path, sine=True, filename='plot_4d.png')

    run_exp(small_path, ks=[1, 2, 5, 10, 20], filename='plot_4e.png')

    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
