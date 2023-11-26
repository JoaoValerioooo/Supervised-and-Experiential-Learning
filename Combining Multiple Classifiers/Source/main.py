from Preprocessing import PREPROCESS
from RANDOM_FOREST_CLASS import RANDOM_FOREST
from DECISION_FOREST_CLASS import DECISION_FOREST
import math
from tabulate import tabulate

# User Input
print('Insert the path in which the folder Data of the project is located.\nExample: /Users/joaovalerio/Documents/MAI UPC/2 Semester/SEL/W2')
file_path = input('-> ')

# Open file
f = open(file_path + '/Data/features_relevance.txt', "w")
f.write("Supervised and Experiential Learning\nPractical Work 2: Combining Multiple Classifiers\nJoão Valério\njoao.agostinho@estudiantat.upc.edu\n04/05/2023\n\n\n")

# Create empty lists to store the results
results_rf = []
results_df = []

# Models
for file, size in zip(['/hepatitis.arff', '/tic-tac-toe.csv', '/CureThePrincess.csv'], ['Small', 'Medium', 'Large']):

    # Information
    print('\n################################################################')
    f.write('\n################################################################')
    print('Dataset File: ' + file)
    f.write('\nDataset File: {}'.format(file))
    print('Dataset Size: ' + size)
    f.write('\nDataset Size: {}'.format(size))
    print('################################################################')
    f.write('\n################################################################\n')

    # Preprocessing the data
    df_train, df_test = PREPROCESS(f).preprocessDataset(file_path + '/Data' + file)
    num_features = len(df_train.columns) - 1

    # Running the models
    for D in [2, 10, 30, 100]:
        for NT in [1, 10, 25, 50, 75, 100]:
            for F_RF, F_DF in zip([1, 2, int(math.log2(num_features + 1)), int(math.sqrt(num_features))],
                                  [int(num_features/4), int(num_features/2), int((3*num_features)/4), -1]):

                ########################### Random Forest ###########################
                # Train the Random Forest
                RF_model = RANDOM_FOREST(D=D, F=F_RF, NT=NT, bootstrapp=0.3, f=f)
                trees = RF_model.train(df_train)
                # Test the Random Forest
                y_pred = RF_model.predict(df_test.drop("Class", axis=1))
                total_accuracy, class_0_accuracy, class_1_accuracy = RF_model.get_accuracy(y_pred, df_test["Class"])
                RF_model.feature_list(trees)
                # Append results to table
                results_rf.append([file, size, D, NT, F_RF, total_accuracy, class_0_accuracy, class_1_accuracy])
                ########################### Random Forest ###########################

                ######################### Decision Forest ###########################
                # Train the Decision Forest
                DF_model = DECISION_FOREST(D=D, F=F_DF, NT=NT, f=f)
                trees = DF_model.train(df_train)
                # Test the Decision Forest
                y_pred = DF_model.predict(df_test.drop("Class", axis=1))
                total_accuracy, class_0_accuracy, class_1_accuracy = DF_model.get_accuracy(y_pred, df_test["Class"])
                DF_model.feature_list(trees)
                # Append results to table
                results_df.append([file, size, D, NT, F_DF, total_accuracy, class_0_accuracy, class_1_accuracy])
                ######################### Decision Forest ###########################

    # Information
    print('\n################################################################')
    f.write('\n################################################################')
    print('################################################################')
    f.write('\n################################################################\n\n\n')

# Close file
f.close()

# Open file
f = open(file_path + '/Data/accuracy.txt', "w")
f.write("Supervised and Experiential Learning\nPractical Work 2: Combining Multiple Classifiers\nJoão Valério\njoao.agostinho@estudiantat.upc.edu\n04/05/2023\n\n\n")

# Sort the results by accuracy (total_accuracy) in descending order for each dataset
results_rf.sort(key=lambda x: (x[0], x[5]), reverse=True)
results_df.sort(key=lambda x: (x[0], x[5]), reverse=True)

# Create tables using tabulate
headers = ["Dataset", "Size", "D", "NT", "F", "Total Accuracy", "Class 0 Accuracy","Class 1 Accuracy"]
table_rf = tabulate(results_rf, headers, tablefmt="grid")
table_df = tabulate(results_df, headers, tablefmt="grid")

# Print the tables
print("Random Forest Results:")
f.write('\nRandom Forest Results:\n')
print(table_rf)
f.write(table_rf)
print("\nDecision Forest Results:")
f.write('\nDecision Forest Results:\n')
print(table_df)
f.write(table_df)

# Close file
f.close()