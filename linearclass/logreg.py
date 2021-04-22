import numpy as np
import util
import math

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    predictions = clf.predict(x_eval)

    # Vars for confustion matrix to calculate a ccuracy and precision
    posPredictedPos = 0
    posPredictedNeg = 0
    negPredictedNeg = 0
    negPredictedPos = 0

    num_eval_data = y_eval.shape[0]
    for i in range(0, num_eval_data):
        if y_eval[i] == 1.0 and predictions[i] == 1.0:
            posPredictedPos += 1
        if y_eval[i] == 1.0 and predictions[i] == 0.0:
            posPredictedNeg += 1
        if y_eval[i] == 0.0 and predictions[i] == 0.0:
            negPredictedNeg += 1
        if y_eval[i] == 0.0 and predictions[i] == 1.0:
            negPredictedPos += 1


    accuracy = (posPredictedPos + negPredictedNeg) / num_eval_data
    precision = posPredictedPos/ (posPredictedPos + negPredictedPos)
    recall = posPredictedPos /  ( posPredictedPos + posPredictedNeg)

    # Write statistics out
    with open(save_path.replace(".txt", "_statistics.txt"), 'w') as f:
        print("==========Statistics ==========", file=f)
        print("accuracy = " + str(accuracy), file=f)
        print("precision = " + str(precision), file=f)
        print("recall = " + str(recall), file=f)


    # Plot decision boundary on top of validation set set
    util.plot(x_eval, y_eval, clf.theta, save_path.replace(".txt", ".png"), correction=1.0)

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, predictions)

    # *** END CODE HERE ***

# Sigmoid Function
def sigmoid(x):
    sig = 1.0 / (1.0 + math.exp(-x))
    return sig

# L1 Normalization
def L1Norm(x):
    sum = 0.0

    for i in range (0, x.shape[0]):
        for j in range (0, x.shape[1]):
            sum += abs(x[i,j])
    return sum



class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1.0, max_iter=1000000, eps=1e-5,
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
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        num_data = x.shape[0]
        num_features = x.shape[1]
        gradient = np.zeros((num_features, 1))

        hessian = np.zeros((num_features, num_features))
        self.theta = np.zeros((num_features, 1))
        h_Theta_X = 0.0
        convergance = False
        iteration = 0

        while not convergance:

            loss = 0.0

            for i in range(0, num_data):

                x_vector = np.transpose(np.array(x[i]).reshape(1, num_features))

                h_Theta_X = sigmoid(np.matmul(np.transpose(x_vector),self.theta))

                if self.verbose is True:
                    loss -= y[i] * np.log(h_Theta_X) + (1 - y[i]) * np.log(1 - h_Theta_X)

                for j in range(0, num_features):
                    gradient[j] -= (y[i] - h_Theta_X) * x[i][j]

                    for k in range(0, num_features):
                        hessian[j][k] -= (h_Theta_X * (1 - h_Theta_X)) * x[i][j] * x[i][k]

            # End of loop over all data points

            # Normalize the aggregates we calculated in the loop over all data.
            gradient = gradient / num_data
            hessian = hessian / num_data
            loss = loss / num_data

            # save the results of theta so we can calculate the L1Norm value of the delta in theta
            new_theta = self.theta + self.step_size * np.matmul(np.linalg.inv(hessian), gradient)
            delta_theta = self.theta  - new_theta

            if L1Norm(delta_theta) < self.eps:
                convergance = True

            self.theta  = new_theta

            iteration = iteration + 1


            # Calculate final loss and print
            if self.verbose is True:
                print("Iteration  = ", iteration)
                loss = -1 * loss/num_data
                print("Loss  = ", loss)

            print("======================================")


        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        num_data = x.shape[0]
        num_features = x.shape[1]
        predictions = np.zeros((num_data, 1))
        h_Theta_X = 0.0

        for i in range (0, num_data):
            x_vector = np.transpose(np.array(x[i]).reshape(1, num_features))
            h_Theta_X = sigmoid(np.matmul(np.transpose(x_vector),self.theta))
            if (h_Theta_X > .5):
                predictions[i] = 1.0

        return predictions
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')

