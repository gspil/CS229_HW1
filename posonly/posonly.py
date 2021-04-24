import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


# Function to predict alpha. Take the set of Y labels and predictions.
# Sum up the predicted value and divide by the number of positive lables.
def predict_alpha(Y, predictions):

    num_data = Y.shape[0]
    sum = 0.0
    num_pos = 0

    for i in range(0, num_data):
        if Y[i] > 0.0 :
            sum += predictions[i]
            num_pos += 1

    return sum/num_pos

def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***

    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_valid, t_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    x_test, y_test   = util.load_dataset(test_path, label_col='y', add_intercept=True)
    x_test, t_test   = util.load_dataset(test_path, label_col='t', add_intercept=True)
    
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    clf_a = LogisticRegression()
    clf_a.fit(x_train, t_train)
    
    predictions = clf_a.predict(x_test)

    np.savetxt(output_path_true, predictions)

    util.plot(x_test, t_test, clf_a.theta, "5A.png", correction=1.0)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    clf_b = LogisticRegression()
    clf_b.fit(x_train, y_train)

    predictions = clf_b.predict(x_test)

    np.savetxt(output_path_naive, predictions)

    util.plot(x_test, t_test, clf_b.theta, "5B.png", correction=1.0)


    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted

    # Run CLF on the valid data set to generate the predictions on valid.
    # Use the predictions and y data to calculate alpha.
    clf_f = LogisticRegression()
    clf_f.fit(x_train, t_train)

    predictions_for_valid = clf_f.predict(x_valid)

    alpha = predict_alpha(t_valid, predictions_for_valid)

    # Run the CLF from part b on test data

    predictions = clf_f.predict(x_test)
   
    scaled_predictions = predictions * alpha

    np.savetxt(output_path_adjusted, scaled_predictions)

    util.plot(x_test, y_test, clf_f.theta, "5F.png", correction=alpha)

    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
