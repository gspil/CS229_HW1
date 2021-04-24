import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    # *** START CODE HERE ***
    # Train a GDA classifier
    gda = GDA()
    gda.fit(x_train, y_train)
    predictions = gda.predict(x_valid)

    # Plot decision boundary on validation set
    util.plot(x_valid, y_valid, gda.theta, save_path.replace(".txt", ".png"), correction=1.0)

    # Use np.savetxt to save predictions on eval set to save_path
    np.savetxt(save_path, predictions)

    # Vars for confusion matrix to calculate accuracy and precision
    posPredictedPos = 0
    posPredictedNeg = 0
    negPredictedNeg = 0
    negPredictedPos = 0

    num_eval_data = y_valid.shape[0]
    for i in range(0, num_eval_data):
        if y_valid[i] == 1.0 and predictions[i] == 1.0:
            posPredictedPos += 1
        if y_valid[i] == 1.0 and predictions[i] == 0.0:
            posPredictedNeg += 1
        if y_valid[i] == 0.0 and predictions[i] == 0.0:
            negPredictedNeg += 1
        if y_valid[i] == 0.0 and predictions[i] == 1.0:
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


    # *** END CODE HERE ***

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1.0, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0 # use this to hold theta and theta0 for export to outside 
        self.theta_N = theta_0 # use this for theta from calculations
        self.theta_0 = None
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose


    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters

        num_data = x.shape[0]
        num_features = x.shape[1]
        self.theta = np.zeros((num_features + 1, 1))
        self.theta_N = np.zeros((num_features, 1))
        self.theta_0  = 0

        mu_0 = np.zeros((num_features, 1))
        mu_0_denominator = 0
        mu_0_numerator = np.zeros((num_features, 1))
        mu_1 = np.zeros((num_features, 1))
        mu_1_denominator = 0
        mu_1_numerator  = np.zeros((num_features, 1))
        phi = 0
        sigma = np.zeros((num_features, num_features))

        # calculate mu_0, mu_1 and phi
        for i in range(0, num_data):
            x_vector = np.transpose(np.array(x[i]).reshape(1, num_features))

            if y[i] == 1 :
                phi += 1
                mu_1_numerator += x_vector
                mu_1_denominator += 1
            else:
                mu_0_numerator += x_vector
                mu_0_denominator += 1

        # Update MUs and phi
        mu_0 = mu_0_numerator / mu_0_denominator
        mu_1 = mu_1_numerator / mu_1_denominator
        phi /= num_data

        # calculate sigma
        for i in range(0, num_data):
            x_vector = np.transpose(np.array(x[i]).reshape(1, num_features))

            if y[i] == 1 :
                class_mu = mu_1
            else:
                class_mu = mu_0

            sigma += np.matmul((x_vector - class_mu), np.transpose(x_vector - class_mu))

        sigma /= num_data
        sigma_inv = np.linalg.inv(sigma)

        #calculate theta 0
        self.theta_0 = 1/2 * ( np.matmul(np.matmul(np.transpose(mu_0), sigma_inv), mu_0)
        - np.matmul(np.matmul(np.transpose(mu_1), sigma_inv), mu_1) )
        - np.log((1 - phi) / phi)

        self.theta_N = -np.matmul(np.transpose(mu_0 - mu_1), sigma_inv)

        # put theta and theta0 back in vector form
        self.theta_N = np.transpose(np.array(self.theta_N).reshape(1, num_features))

        # fill in self.theta for use by client
        self.theta[0] = self.theta_0;
        for i in range (0, num_features) :
            self.theta[i+1] = self.theta_N[i]

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

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

            prediction = 1 / (1 + np.exp(-1 * (np.matmul(np.transpose(self.theta_N), x_vector) + self.theta_0)))

            if (prediction > .5):
                predictions[i] = 1.0
        
        return predictions
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
