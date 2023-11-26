import itertools
import os
import numpy as np
import time
import psutil


class RULES:

    def __init__(self, f):
        # Rules
        self.rules = []  # list to hold the rules discovered by the algorithm
        self.numb_rules = None  # variable to hold the number of rules
        self.precision_rules = []  # list to hold the precision of each rule
        self.coverage_rules = []  # list to hold the coverage of each rule
        self.recall_rules = []  # list to hold the recall of each rule

        # Data Information
        self.numb_classes = None  # number of classes in the dataset
        self.numb_features = None  # number of features in the dataset
        self.name_features = []  # list to hold the names of the features in the dataset

        # Train Information
        self.train_numb_instances = None  # number of instances in the training set
        self.train_time = None  # time taken to train the algorithm
        self.train_mem = None  # memory used to train the algorithm
        self.train_accuracy = None  # accuracy of the training set

        # Test Information
        self.test_numb_instances = None  # number of instances in the test set
        self.test_time = None  # time taken to test the algorithm
        self.test_mem = None  # memory used to test the algorithm
        self.test_accuracy = None  # accuracy of the test set

        # Output File
        self.f = f

#################################################################################
#                               AUXILIAR METHODS                                #
#################################################################################

    def data_information(self, df):
        """
        Set the number of classes, features, and name of features, and number of instances for the input dataset.

        Parameters:
        df (pandas.DataFrame): the input dataset.
        """
        self.numb_classes = df['Class'].nunique() # Get the number of unique classes in the 'Class' column.
        self.numb_features = len(df.columns) - 1 # Get the number of columns in the dataset (excluding the 'Class' column).
        self.name_features = list(df.columns) # Get a list of all column names in the dataset.
        self.train_numb_instances = len(df.index) # Get the number of rows (instances) in the dataset.

    def calculate_total_acc(self, df):
        """
        Calculate the total accuracy for the input dataset.

        Parameters:
        df (pandas.DataFrame): the input dataset.

        Returns:
        float: the total accuracy of the input dataset.
        """
        # Count the number of instances where the predicted class matches the actual class
        num_correct = sum(np.where(df["Class"] == df["Predicted_Class"], True, False))
        # Calculate the total accuracy by dividing the number of correct instances by the total number of instances and multiplying by 100
        total_acc = num_correct * 100 / df.shape[0]
        return total_acc

    def df_subset_creator(self, df, comb, cl):
        """
        Create a subset of the input dataset based on the specified columns and combinations.

        Parameters:
        df (pandas.DataFrame): the input dataset.
        cl (list): a list of column indexes.
        comb (list): a list of combinations.

        Returns:
        pandas.DataFrame: a subset of the input dataset.
        """
        # Create a boolean mask for the first column index and combination
        mask = df[self.name_features[cl[0]]] == comb[0]
        # Iterate over the remaining column indexes and combinations and update the mask
        for i in range(1, len(cl)):
            mask &= df[self.name_features[cl[i]]] == comb[i]
        # Return the subset of the DataFrame that satisfies the mask
        return df[mask].copy()

    def unclassified_data_updater(self, df, rules):
        """
        Update the input dataset by removing the rows that are covered by the generated rules.

        Parameters:
        df (pandas.DataFrame): the input dataset.
        rules (list): a list of generated rules.

        Returns:
        pandas.DataFrame: the updated dataset.
        """
        # Iterate over each rule in the list of rules
        for rule in rules:
            # Initialize the filter for the rule to None
            rule_filter = None
            # Iterate over each antecedent in the rule
            for antecedent in rule:
                # If this is the first antecedent in the rule, create a new filter
                if rule_filter is None:
                    rule_filter = (df[antecedent[0]] == antecedent[1])
                # If this is not the first antecedent, add the condition to the existing filter
                else:
                    rule_filter &= (df[antecedent[0]] == antecedent[1])
            # Remove the rows that satisfy the conditions in the rule
            df = df[~rule_filter]
        # Return the updated dataframe after all the rules have been applied
        return df

    def assess_redudancy(self, auxiliar):
        """
        Check whether the input rule is redundant or not.

        Parameters:
        auxiliar (list): a list of rule antecedents and consequent.

        Returns:
        bool: True if the input rule is redundant, False otherwise.
        """
        # create a set of antecedents and consequent
        candidate_set = set(auxiliar[:-1])
        # loop over the previously discovered rules
        for rule in self.rules:
            # check if the candidate rule's antecedents are a subset of a previously discovered rule's antecedents
            if candidate_set.issubset(set(rule[:-1])):
                return True  # the candidate rule is redundant
        return False  # the candidate rule is not redundant

    def ruler(self, cl, comb, class_val):
        """
        Create a rule given a combination of column indexes, column values and the class value.

        Parameters:
        cl (list): a list of column indexes.
        comb (list): a list of combinations.
        class_val (str): the class value.

        Returns:
        list: a list that represents the rule, containing the antecedents and the consequent.
        """
        # Create a list of tuples that represent the antecedents of the rule
        # by iterating over the list of column indexes and combinations,
        # and zipping them with the names of the corresponding features.
        auxiliar = list(zip([self.name_features[cl[i]] for i in range(len(cl))], comb))
        # Add a tuple to the list of antecedents that represents the consequent
        # of the rule, containing the name of the class feature and its value.
        auxiliar.append(tuple(["Class", class_val]))
        # Return the list of antecedents and consequent, which represents the rule.
        return auxiliar

#################################################################################
#                               RULE METHODS                                    #
#################################################################################

    def single_ruler(self, df):
        """
        Discover single antecedent rules.

        Parameters:
        df (pandas.DataFrame): the input dataset.

        Returns:
        list: a list of single antecedent rules discovered.
        """
        # Iterate over all features except the last (which is assumed to be the target class).
        for feature in self.name_features[:-1]:
            # Iterate over all unique values for the feature.
            for value in df[feature].unique():
                # Extract the subset of the DataFrame where the feature has the given value.
                df_subset = df.loc[df[feature] == value]
                # Check if all instances in the subset have the same class value.
                rule_val, class_val = (df_subset["Class"].iloc[0] == df_subset["Class"]).all(), df_subset["Class"].iloc[0]
                # If the subset satisfies the rule, create a new rule.
                if rule_val:
                    # Create a tuple for the antecedent and consequent of the rule.
                    antecedent = tuple([feature, value])
                    consequent = tuple(["Class", class_val])
                    # Create a boolean mask for the instances that satisfy the antecedent.
                    antecedent_mask = df[antecedent[0]] == antecedent[1]
                    # Count the number of instances that satisfy the antecedent and consequent, and only the antecedent.
                    num_satisfy_antecedent_and_consequent = (df[consequent[0]][antecedent_mask] == consequent[1]).sum()
                    num_satisfy_antecedent = antecedent_mask.sum()
                    # Calculate the precision, recall, and coverage of the rule.
                    precision = (num_satisfy_antecedent_and_consequent / num_satisfy_antecedent) * 100
                    recall = (num_satisfy_antecedent_and_consequent / (df["Class"] == class_val).sum()) * 100
                    coverage = df_subset.shape[0] * 100 / self.train_numb_instances
                    # Add the precision, recall, and coverage to the corresponding lists.
                    self.precision_rules.append(precision)
                    self.recall_rules.append(recall)
                    self.coverage_rules.append(coverage)
                    # Add the rule to the list of rules.
                    self.rules.append([antecedent, consequent])
                    # Print the rule with its precision, recall, and coverage.
                    print("IF {0} = {1} THEN Class = {2} WITH Coverage {3:.2f}% & Recall {4:.2f}% & Precision {5:.2f}%\n".format(feature, value, class_val, coverage, recall, precision))
                    self.f.write("IF {0} = {1} THEN Class = {2} WITH Coverage {3:.2f}% & Recall {4:.2f}% & Precision {5:.2f}%\n".format(feature, value, class_val, coverage, recall, precision))
        # Update the unclassified DataFrame with the new rules.
        return self.unclassified_data_updater(df, self.rules)

    def non_single_ruler(self, df, unclassified):
        """
        Discover rules with more than one antecedent.

        Parameters:
        df (pandas.DataFrame): the input dataset.
        unclassified (pandas.DataFrame): a dataset containing unclassified instances.

        Returns:
        pandas.DataFrame: the updated unclassified dataset.
        """
        # Loop through different numbers of features to generate rules for
        for i in range(2, self.numb_features):
            rules, combs = [], []
            values = [list(np.unique(np.array(unclassified.iloc[:, attrib]))) for attrib in range(self.numb_features)]
            # Create all possible combinations of columns for this number of features
            ids = list(itertools.combinations(range(self.numb_features), i))
            for id in ids:
                # Create all possible combinations of values for each column combination
                combs.append(list(itertools.product(*[values[attrib] for attrib in id])))
            # Loop through each column combination and value combination and generate rules
            for id, cl in enumerate(ids):
                for comb in combs[id]:
                    # Create a subset of the input dataset based on the current column and value combinations
                    df_subset = self.df_subset_creator(df, comb, cl)
                    if not df_subset.empty:
                        # Determine the most common class value in the subset
                        class_val = df_subset["Class"].mode().values[0]
                        # Check if all instances in the subset belong to the most common class
                        rule_val = (df_subset["Class"] == class_val).all()
                        if rule_val:
                            # Generate a rule based on the column and value combinations and the most common class
                            auxiliar = self.ruler(cl, comb, class_val)
                            # Check if the rule is redundant
                            if not self.assess_redudancy(auxiliar):
                                # Calculate precision, recall, and coverage for the rule
                                rl = [f"{auxiliar[i][0]} = {auxiliar[i][1]}" for i in range(len(auxiliar))]
                                true_positives = df_subset[df_subset["Class"] == class_val].shape[0]
                                false_negatives = df_subset[df_subset["Class"] != class_val].shape[0]
                                false_positives = self.unclassified_data_updater(df_subset, [auxiliar]).query("Class == @class_val").shape[0]
                                precision = true_positives / (true_positives + false_positives) * 100
                                recall = true_positives / (true_positives + false_negatives) * 100
                                coverage = df_subset.shape[0] * 100 / self.train_numb_instances
                                # Append precision, recall, coverage, and rule to corresponding lists
                                self.precision_rules.append(precision)
                                self.recall_rules.append(recall)
                                self.coverage_rules.append(coverage)
                                self.rules.append(auxiliar)
                                rules.append(auxiliar)
                                # Print the generated rule along with its coverage, recall, and precision
                                print(f"IF {' & '.join(rl[:-1])} THEN Class = {class_val} WITH Coverage {coverage:.4f}% & Recall {recall:.2f}% & Precision {precision:.2f}%\n")
                                self.f.write(f"IF {' & '.join(rl[:-1])} THEN Class = {class_val} WITH Coverage {coverage:.4f}% & Recall {recall:.2f}% & Precision {precision:.2f}%\n")
            # Update the unclassified dataset based on the generated rules
            unclassified = self.unclassified_data_updater(unclassified, rules)
            if unclassified.empty: break
        return unclassified

    def unclassified_ruler(self, df, unclassified):
        """
        Create a rule for the unclassified instances.

        Parameters:
        df (pandas.DataFrame): The input train dataset.
        unclassified (pandas.DataFrame): A dataset containing unclassified instances.
        """
        for _, instance in unclassified.iterrows():
            aux_rule = list(zip(unclassified.columns, instance))
            rule = [str(aux_rule[i][0]) + " = " + str(aux_rule[i][1]) for i in range(len(aux_rule))]
            # Calculate the coverage of the rule
            rule_coverage = (sum(df.apply(lambda row: all(row[feature] == value for (feature, value) in aux_rule),axis=1)) / self.train_numb_instances) * 100
            # Calculate the number of instances satisfying the antecedent of the rule
            antecedent_count = sum(df.apply(lambda row: all(row[feature] == value for (feature, value) in aux_rule[:-1]), axis=1))
            # Calculate the number of instances satisfying both the antecedent and consequent of the rule
            consequent_count = sum(df.apply(lambda row: all(row[feature] == value for (feature, value) in aux_rule), axis=1))
            # Calculate the number of instances in the same class label that R is classifying
            same_class_count = sum(df[df[unclassified.columns[-1]] == instance[-1]].count())
            # Calculate the recall of the rule
            if same_class_count == 0:rule_recall = 0
            else:rule_recall = (consequent_count / same_class_count)*100
            # Calculate the precision of the rule
            if antecedent_count == 0:rule_precision = 0
            else:rule_precision = (consequent_count / antecedent_count)*100
            print("IF {0} THEN {1} WITH Coverage = {2:.2f}% & Recall = {3:.2f}% & Precision = {4:.2f}% \n".format(" & ".join(rule[:-1]), rule[-1], rule_coverage, rule_recall, rule_precision))
            self.f.write("IF {0} THEN {1} WITH Coverage = {2:.2f}% & Recall = {3:.2f}% & Precision = {4:.2f}% \n".format(" & ".join(rule[:-1]), rule[-1], rule_coverage, rule_recall, rule_precision))
            self.rules.append(aux_rule)
            self.precision_rules.append(rule_precision)
            self.recall_rules.append(rule_recall)
            self.coverage_rules.append(rule_coverage)

#################################################################################
#                               TRAIN METHODS                                   #
#################################################################################

    def fit (self, df):
        """
        Train the model on the input dataframe.

        Parameters:
        df (pd.DataFrame): Input dataframe to train the model.
        """
        # Time
        t_0 = time.time()
        # Memory in kB
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024
        # Train the model
        self._fit(df)
        # Memory in kB
        self.train_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024) - mem_start
        # Time
        self.train_time = time.time() - t_0
        # Number of rules
        self.numb_rules = len(self.rules)
        # Accuracy
        self.train_accuracy = self.calculate_total_acc(self._predict(df))
        # Print Train Information
        print("Train Process Completed Successfully.")
        self.f.write("\nTrain Process Completed Successfully.\n")
        self.get_train_info()

    def _fit(self, df):
        """
        Private function to fit the model on the input dataframe.

        Parameters:
        df (pd.DataFrame): Input dataframe to train the model.
        """
        # Get Data Information
        self.data_information(df)
        # Ruler Creator -> selector = 1
        unclassified = self.single_ruler(df)
        # Ruler Creator -> 1 < selector <= numb_features
        unclassified = self.non_single_ruler(df, unclassified)
        # Ruler Creator (unclassified instances) -> selector = numb_features
        if not unclassified.empty: self.unclassified_ruler(df, unclassified)

#################################################################################
#                               TEST METHODS                                    #
#################################################################################

    def predict(self, df):
        """
        Predict the output for the input dataframe.

        Parameters:
        df (pd.DataFrame): Input dataframe to predict the output.
        """
        # Get the number of instances
        self.test_numb_instances = len(df.index)
        # Time
        t_0 = time.time()
        # Memory in kB
        mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024
        # Test the model
        df = self._predict(df)
        # Memory in kB
        self.test_mem = (psutil.Process(os.getpid()).memory_info().rss / 1024) - mem_start
        # Time
        self.test_time = time.time() - t_0
        # Accuracy
        self.test_accuracy = self.calculate_total_acc(df)
        # Print Test Information
        print("\nTest Process Completed Successfully.")
        self.f.write("\n\nTest Process Completed Successfully.\n")
        self.get_test_info()
        return self.test_accuracy

    def _predict(self, df):
        """
        Private function to predict the output for the input dataframe.

        Parameters:
        df (pd.DataFrame): Input dataframe to predict the output.

        Returns:
        pd.DataFrame: Dataframe containing predicted class for each instance.
        """
        predicted_class = []
        for _, instance in df.iterrows():
            # Check if the instance matches any rule
            for rule in self.rules:
                check_list = [instance[feature] == value for feature, value in rule[:-1]]
                # If all the conditions in the rule are True, append the predicted class to the list
                if all(check_list):
                    predicted_class.append(rule[-1][1])
                    break
            # If no rule is found for the instance, append NaN
            else: predicted_class.append(np.nan)
        # Add the predicted class column to the dataframe and return it
        df["Predicted_Class"] = predicted_class
        return df

#################################################################################
#                            INFORMATION METHODS                                #
#################################################################################

    def get_train_info(self):
        """
        Print the information regarding the train process.
        """
        print('\nTrain Information:\n')
        self.f.write('\nTrain Information:\n')
        print('Number of Features:', self.numb_features)
        self.f.write('\nNumber of Features: {}'.format(self.numb_features))
        print('Name of Features:', self.name_features[:-1])
        self.f.write('\nName of Features: {}'.format(self.name_features[:-1]))
        print('Number of Instances:', self.train_numb_instances)
        self.f.write('\nNumber of Instances: {}'.format(self.train_numb_instances))
        print('Number of Classes:', self.numb_classes)
        self.f.write('\nNumber of Classes: {}'.format(self.numb_classes))
        print('Number of Rules:', self.numb_rules)
        self.f.write('\nNumber of Rules: {}'.format(self.numb_rules))
        print('Train Time [s]:', self.train_time)
        self.f.write('\nTrain Time [s]: {}'.format(self.train_time))
        print('Train Memory [kB]:', self.train_mem)
        self.f.write('\nTrain Memory [kB]: {}'.format(self.train_mem))
        print('Train Accuracy:', self.train_accuracy)
        self.f.write('\nTrain Accuracy: {}'.format(self.train_accuracy))

    def get_test_info(self):
        """
        Print the information regarding the test process.
        """
        print('\nTest Information:\n')
        self.f.write('\nTest Information:\n')
        print('Number of Features:', self.numb_features)
        self.f.write('\nNumber of Features: {}'.format(self.numb_features))
        print('Name of Features:', self.name_features[:-1])
        self.f.write('\nName of Features: {}'.format(self.name_features[:-1]))
        print('Number of Instances:', self.test_numb_instances)
        self.f.write('\nNumber of Instances: {}'.format(self.test_numb_instances))
        print('Number of Classes:', self.numb_classes)
        self.f.write('\nNumber of Classes: {}'.format(self.numb_classes))
        print('Number of Rules:', self.numb_rules)
        self.f.write('\nNumber of Rules: {}'.format(self.numb_rules))
        print('Test Time [s]:', self.test_time)
        self.f.write('\nTest Time [s]: {}'.format(self.test_time))
        print('Test Memory [kB]:', self.test_mem)
        self.f.write('\nTest Memory [kB]: {}'.format(self.test_mem))
        print('Test Accuracy:', self.test_accuracy)
        self.f.write('\nTest Accuracy: {}'.format(self.test_accuracy))