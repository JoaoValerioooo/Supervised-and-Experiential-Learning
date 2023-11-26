from CART_CLASS import CART
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from collections import defaultdict
from tabulate import tabulate

class RANDOM_FOREST:

    def __init__(self, D = 10, F = 1, NT = 1, bootstrapp = 0.2, f = None):
        """
        Constructor for the Random Forest class.

        Parameters:
        - D (int): Max depth of the decision trees in the forest (default = 10)
        - F (int): Number of features to consider for split at each node (default = 1)
        - NT (int): Number of trees in the forest (default = 1)
        - bootstrapp (float): Percentage of the training data to use for bootstrapped sampling (default = 0.2)
        - f (file): File object to write the output to (default = None)
        """
        self.D = D if D > 0 else 1
        self.F = F if F > 0 else 1
        self.NT = NT if NT > 0 else 1
        self.bootstrapp = bootstrapp if 0 < bootstrapp <= 1 else 0.2
        self.trees = []
        self.CART_models = []
        self.f = f

    #################################################################################
    #                                    MAIN METHOD                                #
    #################################################################################

    def train(self, df_train):
        """
        Train the Random Forest using the training dataset.

        Parameters:
        - df_train (DataFrame): Training dataset

        Returns:
        - List: List of decision trees in the forest
        """
        for tree_id in range(0, self.NT):
            # Create a CART model
            model = CART(D=self.D, F = self.F)
            self.CART_models.append(model)
            # Bootstrapped sampling of the original training set
            df_bootstrapped = df_train.sample(frac = self.bootstrapp, replace=False)
            df_bootstrapped.reset_index(drop=True, inplace=True)
            # CART obtains the tree from the bootstrapped sample
            model.train(df_bootstrapped)
            tree = model.get_tree()
            self.trees.append(tree)
        return self.get_trees()

    #################################################################################
    #                                AUXILIAR METHOD                                #
    #################################################################################

    def get_trees(self):
        """
        Get the list of decision trees in the forest.

        Returns:
        - List: List of decision trees in the forest
        """
        return self.trees

    #################################################################################
    #                             INTERPRETER METHOD                                #
    #################################################################################

    def predict(self, X_test):
        """
        Make predictions using the trained Random Forest.

        Parameters:
        - X_test (DataFrame): Test dataset

        Returns:
        - List: List of predicted labels for each sample in the test dataset
        """
        # Prediction for each tree
        predictions = [self.CART_models[tree_id].predict(X_test) for tree_id in range(0, self.NT)]
        # Guaranteeing the absense of None predictions
        final_predictions = []
        for id in range(len(predictions)):
            final_predictions.append([pred if pred is not None else 1 for pred in predictions[id]])
        # Determine the majority voted class for each position
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=np.array(final_predictions)).tolist()

    #################################################################################
    #                             INFORMATION METHODS                               #
    #################################################################################

    def get_accuracy(self, y_pred, y_test):
        """
        Calculate the accuracy, class 0 accuracy, and class 1 accuracy of the trained Random Forest classifier.

        Parameters:
        - y_pred (list): List of predicted class labels.
        - y_test (list): List of true class labels.

        Returns:
        - Total accuracy, class 0 accuracy, and class 1 accuracy
        """
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        # Calculate accuracies
        total_accuracy = accuracy_score(y_test, y_pred)
        class_0_accuracy = tn / (tn + fp)
        class_1_accuracy = tp / (tp + fn)
        print('\nRandom Forest:')
        self.f.write('\n\nRandom Forest:')
        self.get_characteristics()
        print('Accuracy: ', total_accuracy)
        self.f.write('\nAccuracy: {}'.format(total_accuracy))
        print('Class 0 Accuracy: ', class_0_accuracy)
        self.f.write('\nClass 0 Accuracy: {}'.format(class_0_accuracy))
        print('Class 1 Accuracy: ', class_1_accuracy)
        self.f.write('\nClass 1 Accuracy: {}'.format(class_1_accuracy))
        return total_accuracy, class_0_accuracy, class_1_accuracy

    def get_characteristics(self):
        """
        Indicate the Hyper-parameters of the model.
        """
        print('Max Depth: ', self.D)
        self.f.write('\nMax Depth: {}'.format(self.D))
        print('Number of Trees: ', self.NT)
        self.f.write('\nNumber of Trees: {}'.format(self.NT))
        print('Number of Features: ', self.F)
        self.f.write('\nNumber of Features: {}'.format(self.F))
        print('Percentage of Bootstrapp: ', self.bootstrapp)
        self.f.write('\nPercentage of Bootstrapp: {}'.format(self.bootstrapp))

    def feature_list(self, trees):
        """
        Indicate the feature list relevance.
        """
        # Initialize dictionaries to store frequency, sum of split points, and sum of depths for each feature
        frequency = defaultdict(int)
        sum_split_point = defaultdict(float)
        sum_depth = defaultdict(int)
        # Initialize variables to store class frequency
        class_0_frequency = 0
        class_1_frequency = 0
        # Loop through each tree in the 'trees' variable
        for tree in trees:
            # Define a helper function to recursively traverse the tree and update the frequency, sum of split points, and sum of depths
            def traverse(node, depth):
                nonlocal class_0_frequency, class_1_frequency
                if 'feature' in node:
                    feature = node['feature']
                    frequency[feature] += 1
                    sum_split_point[feature] += node['split_point']
                    sum_depth[feature] += depth
                    traverse(node['left'], depth + 1)
                    traverse(node['right'], depth + 1)
                else:
                    if 'class' in node:
                        if node['class'] == 0: class_0_frequency += 1
                        elif node['class'] == 1: class_1_frequency += 1
            # Start traversing the tree from the root
            traverse(tree, 0)
        # Calculate the average split point and average depth for each feature, truncated to one decimal place
        average_split_point = {feature: round(sum_split_point[feature] / frequency[feature], 1) if frequency[feature] > 0 else 'N/A' for feature in frequency}
        average_depth = {feature: round(sum_depth[feature] / frequency[feature], 1) if frequency[feature] > 0 else 'N/A' for feature in frequency}
        # Sort the features based on the criteria provided
        sorted_features = sorted(frequency.keys(),key=lambda x: (-frequency[x], average_split_point[x], average_depth[x]))
        # Prepare the data for the table
        table_data = []
        for feature in sorted_features:
            if feature != 'class': table_data.append([feature, frequency[feature], average_split_point[feature], average_depth[feature]])
        table_data.append(['class (0)', class_0_frequency, 'N/A', 'N/A'])
        table_data.append(['class (1)', class_1_frequency, 'N/A', 'N/A'])
        table_data.append(['class (total)', class_0_frequency + class_1_frequency, 'N/A', 'N/A'])
        # Print the results as a table using tabulate
        print(tabulate(table_data, headers=['Feature', 'Frequency', 'Avg Split Point', 'Avg Depth'], tablefmt='grid'))
        self.f.write('\n\n')
        self.f.write(tabulate(table_data, headers=['Feature', 'Frequency', 'Avg Split Point', 'Avg Depth'], tablefmt='grid'))