import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import KNNImputer

class PREPROCESS:

    def __init__(self, f):
        self.f = f

    def replaceCategoricalMissings(self, df, missingCode, replaceCode):
        """
        Replaces missing values represented as `missingCode` in categorical columns of a dataframe with `replaceCode`.

        Parameters:
        - df: pandas dataframe
            Input dataframe with missing values
        - missingCode: str
            Code representing the missing values in the dataframe
        - replaceCode: str
            Code to replace the missing values with

        Returns:
        - df: pandas dataframe
            Dataframe with missing values replaced with `replaceCode` in categorical columns
        """
        for col in df.columns:
            # Check if column is categorical and replace missing values with replaceCode in categorical columns
            if df[col].dtype == object: df.loc[df[col] == missingCode, col] = replaceCode
        return df

    def eraseClassColumn(self, df):
        """
        Separates the class column from the input dataframe and applies label encoding to it.

        Parameters:
        - df: pandas dataframe
            Input dataframe with the class column included

        Returns:
        - dfaux: pandas dataframe
            Dataframe with the class column removed
        - labels: numpy array
            Array of labels obtained by applying label encoding to the class column of the input dataframe
        """
        dfaux = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        # Apply label encoding to the labels column
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        return dfaux, labels

    def normalizeDataset(self, df):
        """
        Normalizes the numerical columns of a dataframe between 0 and 1.

        Parameters:
        - df: pandas dataframe
            Input dataframe with numerical columns to be normalized

        Returns:
        - df_norm: pandas dataframe
            Dataframe with the numerical columns normalized between 0 and 1
        """
        df_norm = df.copy()
        for col in df_norm.columns:
            # Check if the column is not of type string and normalize it between 0 and 1
            if not isinstance(df_norm[col][0], str): df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        return df_norm

    def replaceNumericalMissings(self, df):
        """
        Replaces missing values in numerical columns of a dataframe with K-Nearest Neighbor (KNN) imputation.

        Parameters:
        - df: pandas dataframe
            Input dataframe with numerical columns containing missing values

        Returns:
        - df_copy: pandas dataframe
            Dataframe with the missing values replaced with KNN imputation
        """
        numerical = []
        df_copy = df.copy()
        # Loop over columns in dataframe and identify numerical columns
        for ind, col in enumerate(df.columns.values):
            if df[col].dtype != object:
                numerical.append(ind)
        # If no numerical columns, return the original dataframe
        if len(numerical) == 0:
            return df_copy
        # Select numerical columns from the original dataframe
        dd = df.iloc[:, numerical]
        colnames = dd.columns
        # Apply KNN imputation to the selected columns
        imputer = KNNImputer(weights='distance')
        imputer.fit(dd)
        ddarray = imputer.transform(dd)
        ddclean = pd.DataFrame(ddarray, columns=colnames)
        # Replace missing values in original dataframe with imputed values
        for col in ddclean.columns:
            df_copy[col] = ddclean[col]
        return df_copy

    def applyOneHotEncoding(self, df):
        """
        Apply One-Hot Encoding to the categorical features.

        Parameters:
        - df: pandas dataframe
        Input dataframe with numerical and categorical columns

        Returns:
        - df: pandas dataframe
        Dataframe only with numerical columns
        """
        categorical = []
        for col in df.columns:
            if df[col].dtype == object:
                categorical.append(col)
        df = pd.get_dummies(df, columns=categorical)
        return df

    def split(self, df, labels):
        """
        Splits a dataframe and its corresponding labels into training and testing sets.

        Parameters:
        - df: pandas dataframe
            Input dataframe to be split into training and testing sets
        - labels: numpy array
            Array of labels corresponding to the input dataframe

        Returns:
        - df_train: pandas dataframe
            Dataframe containing the training set (80% of the original dataset)
        - df_test: pandas dataframe
            Dataframe containing the testing set (20% of the original dataset)
        """
        # Calculate the index to split the dataframe into training and testing sets
        lim = round(len(labels) * 0.8)
        # Split the dataframe into training and testing sets
        df_train = df[:lim][:]
        df_test = df[lim:][:]
        # Create dataframes for the labels corresponding to the training and testing sets
        labels_train = pd.DataFrame(labels[:lim][:], columns=['Class'])
        labels_test = pd.DataFrame(labels[lim:][:], columns=['Class'])
        # Merge the training and testing sets with their corresponding labels
        df_train = pd.merge(df_train, labels_train, how='right', left_index=True, right_index=True)
        df_test.reset_index(drop=True, inplace=True)
        df_test = pd.merge(df_test, labels_test, how='right', left_index=True, right_index=True)
        return df_train, df_test

    def preprocessDataset(self, filename):
        """
        Preprocesses a dataset by applying a series of data cleaning and transformation functions.

        Parameters:
        - filename: str
            Name of the file containing the dataset to be preprocessed

        Returns:
        - df_train: pandas dataframe
            Dataframe containing the training set
        - df_test: pandas dataframe
            Dataframe containing the testing set
        """
        # Convert data to pandas dataframe
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            df.columns = [*df.columns[:-1], 'class']
            df = df.sample(frac=1, random_state=32).reset_index(drop=True)
        elif filename.endswith('.arff'):
            data = arff.loadarff(filename)
            df = pd.DataFrame(data[0])
            df = df.sample(frac=1, random_state=15).reset_index(drop=True)
            # Convert all object dtype columns to string type
            for col in df.columns:
                if df[col].dtype == object: df[col] = df[col].str.decode('utf-8')
        # Erase class column and return it as labels
        df, labels = self.eraseClassColumn(df)
        # Replace missing values with "Unknown" for categorical columns
        df = self.replaceCategoricalMissings(df, "?", "Unknown")
        # Replace missing values with the column mean for numerical columns
        df = self.replaceNumericalMissings(df)
        # One-Hot Encoding
        df = self.applyOneHotEncoding(df)
        # Normalize the dataset to have zero mean and unit variance
        df = self.normalizeDataset(df)
        # Split dataset into training and testing sets
        df_train, df_test = self.split(df, labels)
        print("\nPre-Processing Completed Successfully.\n")
        self.f.write("\nPre-Processing Completed Successfully.\n\n")
        return df_train, df_test