import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = PoissonRegression(step_size=lr)

    # Fit a Poisson Regression model
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    # *** END CODE HERE ***

# Function to calculate 1 devided by the negative EXP of the log of Theta T  x + Theta0
def poisson_expected_val (theta, x):
    return np.exp(np.matmul(np.transpose(theta), x))
#   return 1 / ( 1 + np.exp(-1 * (np.matmul(np.transpose(theta), x))))
#    return 1 / ( 1 + np.exp(-1 * (np.matmul(np.transpose(theta), x) + theta_0)))

# L1 Normalization
def L1Norm(x):
    sum = 0.0

    for i in range (0, x.shape[0]):
        for j in range (0, x.shape[1]):
            sum += abs(x[i,j])
    return sum

class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.theta_0 = 0.0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        num_data = x.shape[0]
        num_features = x.shape[1]

        convergance = False
        iteration = 0

        hessian = np.zeros((num_features, num_features))
        self.theta = np.zeros((num_features, 1))
        theta_step = np.zeros((num_features, 1))

        #for i in range(0, num_features):
            #self.theta[i] = 0.5

        while not convergance:

            for i in range(0, num_data):

                expected_val = poisson_expected_val(self.theta, x[i])

                for j in range(0, num_features):
                        theta_step[j] += (y[i] - expected_val) * x[i][j]

            #end of iteration over data

            # save the results of update to theta so we can calculate the L1Norm value of the delta in theta
            foo = (self.step_size/num_data) * theta_step

            #new_theta = (self.theta) + ((self.step_size/num_data) * theta_step)
            new_theta = (self.theta) + ((self.step_size) * theta_step)

            delta_theta = self.theta - new_theta

            if L1Norm(delta_theta) < self.eps:
                convergance = True

            # Update theta
            self.theta = new_theta

            if self.verbose is True:
                print("Iteration  = ", iteration)
                iteration += 1
                print("delta = ", L1Norm(delta_theta))

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
