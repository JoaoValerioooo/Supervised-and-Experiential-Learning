import numpy as np
import random

class Node:

    def __init__(self, data, y_col, depth, F):
        # Data information
        self.data = data
        self.y_col = y_col
        self.X = data.drop(y_col, axis=1)
        # Prediction
        self.predicted_label = np.argmax(list(self.counter(data[y_col].to_list()).values()))
        # Hyper-parameters
        self.depth = depth
        self.F = F
        # Node
        self.left = None
        self.right = None

    #################################################################################
    #                                    MAIN METHOD                                #
    #################################################################################

    def spliter(self, D):
        """
        Method to split the data at the node based on the optimal feature and split point.

        Parameters:
        - D: int, the maximum depth of the tree
        """
        self.opt_feature, self.opt_split = self.optimal_split()
        if self.opt_feature is not None:
            if (self.depth < D):
                below, above = self.data[self.data[self.opt_feature] < self.opt_split], self.data[self.data[self.opt_feature] > self.opt_split]
                self.left, self.right = Node(below, 'Class', depth = self.depth + 1, F=self.F), Node(above, 'Class', depth = self.depth + 1, F=self.F)
                self.left.spliter(D)
                self.right.spliter(D)

    #################################################################################
    #                               AUXILIAR METHODS                                #
    #################################################################################

    def counter(self, y):
        """
        Method to count the occurrences of each class label in the target variable.

        Parameters:
        - y: list, the target variable

        Returns:
        - dict, a dictionary with class labels as keys and their counts as values
        """
        return {0: y.count(0), 1: y.count(1)}

    def calculate_gini_index(self, counts):
        """
        Method to calculate the Gini index for a node.

        Parameters:
        - counts: dict, a dictionary with class labels as keys and their counts as values

        Returns:
        - float, the Gini index
        """
        return 1 - ((counts[0] / (counts[0] + counts[1]))**2 + (counts[1] / (counts[0] + counts[1]))**2)

    def get_splits(self, feature):
        """
        Method to get the split points for a given feature.

        Parameters:
        - feature: str, the feature for which to get the split points

        Returns:
        - numpy array, the split points
        """
        return (np.unique(self.X[feature])[1:] + np.unique(self.X[feature])[:-1])/2

    def get_split_gini(self, feature, split_point):
        """
        Method to calculate the Gini index reduction for a given feature and split point.

        Parameters:
        - feature: str, the feature for which to calculate the Gini index reduction
        - split_point: float, the split point for which to calculate the Gini index reduction

        Returns:
        - float, the Gini index reduction
        """
        below, above = self.data[self.data[feature] < split_point], self.data[self.data[feature] > split_point]
        below_counts, above_counts = self.counter(below[self.y_col].to_list()), self.counter(above[self.y_col].to_list())
        below_weight, above_weight = len(below) / (len(below) + len(above)), len(above) / (len(below) + len(above))
        below_gini, above_gini = self.calculate_gini_index(below_counts), self.calculate_gini_index(above_counts)
        return (below_weight*below_gini) + (above_weight*above_gini)

    def optimal_split(self):
        """
        Finds the optimal feature and split point based on the Gini index.

        Returns:
        - opt_feature (str): The name of the feature that results in the optimal split.
        - opt_split (float): The value of the split point that results in the optimal split.
        """
        opt_feature, opt_split, opt_gini_reduction = None, None, 0
        random.seed(123)
        random_features = random.sample(list(self.X.columns), self.F)
        for feature in random_features:
            for split in self.get_splits(feature):
                gini_reduction = self.calculate_gini_index(self.counter(self.data[self.y_col].to_list())) - self.get_split_gini(feature, split)
                if gini_reduction > opt_gini_reduction: opt_gini_reduction, opt_feature, opt_split = gini_reduction, feature, split
        return opt_feature, opt_split