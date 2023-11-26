from Preprocessing import PREPROCESS
from RULES_ALG import RULES

# User Input
print('Insert the path in which the folder Data of the project is located.\nExample: /Users/joaovalerio/Documents/MAI UPC/2 Semester/SEL/W1')
file_path = input('-> ')

# Open file
f = open(file_path + '/Data/rules.txt', "w")
f.write("Supervised and Experiential Learning\nPractical Work 1: Rule Based Classifier RULES\nJoão Valério\njoao.agostinho@estudiantat.upc.edu\n31/03/2023\n\n\n")

# Models
for file, size in zip(['/hepatitis.arff', '/tic-tac-toe.csv', '/phoneme.csv'], ['Small', 'Medium', 'Large']):

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

    # Training the model
    model = RULES(f)
    model.fit(df_train)

    # Testing the model
    model.predict(df_test)

    # Information
    print('\n################################################################')
    f.write('\n################################################################')
    print('################################################################')
    f.write('\n################################################################\n\n\n')

# Close file
f.close()