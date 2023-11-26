from NODE_CLASS import Node

class CART:

    def __init__(self, D = 10, F = 1):
        self.D = D if D > 0 else 1
        self.F = F if F > 0 else 1

    #################################################################################
    #                                    MAIN METHOD                                #
    #################################################################################

    def train(self, train):
        """
        Train the CART algorithm on the provided training data.

        Args:
        train (DataFrame): Training data as a pandas DataFrame.

        Returns:
        None
        """
        self.root = Node(train, "Class", depth=0, F=self.F)
        self.root.spliter(self.D)

    #################################################################################
    #                                 TREE GENERATOR                                #
    #################################################################################

    def get_tree(self):
        """
        Returns the tree structure as a dictionary.

        Returns:
        dict: Tree structure represented as a dictionary.
        """
        return self._get_tree_dict(self.root)

    def _get_tree_dict(self, node):
        """
        Recursive helper function to generate the tree dictionary.

        Args:
        node (Node): Current node of the tree.

        Returns:
        dict: Tree structure represented as a dictionary.
        """
        tree_dict = {}
        if node is not None:
            if node.opt_feature is not None:
                tree_dict['feature'] = node.opt_feature
                tree_dict['split_point'] = node.opt_split
                tree_dict['depth'] = node.depth
                tree_dict['left'] = self._get_tree_dict(node.left)
                tree_dict['right'] = self._get_tree_dict(node.right)
            else:
                tree_dict['class'] = node.predicted_label
        return tree_dict

    #################################################################################
    #                               PREDICTOR METHOD                                #
    #################################################################################

    def instance_predict(self, row):
        """
        Predict the class label for a single instance using the trained CART tree.

        Args:
        row (Series): Single instance to be predicted as a pandas Series.

        Returns:
        str: Predicted class label.
        """
        node = self.root
        while node.depth < self.D:
            split_feature, split_point = node.opt_feature, node.opt_split
            if split_feature is None: break
            else:
                if row[split_feature] < split_point: node = node.left
                else: node = node.right
        return node.predicted_label

    def predict(self, X_test):
        """
        Predict the class labels for a set of instances using the trained CART tree.

        Args:
        X_test (DataFrame): Instances to be predicted as a pandas DataFrame.

        Returns:
        list: Predicted class labels.
        """
        predictions = []
        for idx, row in X_test.iterrows():
            predictions.append(self.instance_predict(row))
        return predictions