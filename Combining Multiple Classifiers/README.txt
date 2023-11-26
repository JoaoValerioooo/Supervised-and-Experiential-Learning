Supervised and Experiential Learning
Practical Work 2: Combining Multiple Classifiers
João Valério
joao.agostinho@estudiantat.upc.edu
04/05/2023

Requirements:
numpy==1.24.2
pandas==1.5.3
scikit_learn==1.2.2
scipy==1.10.1
tabulate==0.9.0

Software:
PyCharm CE

Programming Language:
Python - version 3.8.8

Execution:
To execute the code through the terminal the following steps should be taken:
pip install numpy
pip install pandas
pip install scikit_learn
pip install scipy
pip install tabulate
python3 /PATH_WHERE_THE_FILE_main.py_IS_INSERTED
Ex: /Users/joaovalerio/Documents/"MAI UPC"/"2 Semester"/SEL/W1/source/main.py
/PATH_WHERE_THE_FOLDER_DATA_IS_INSERTED
Ex: /Users/joaovalerio/Documents/MAI UPC/2 Semester/SEL/W1

PW2-SEL-2223-JOAOVALERIO Folder structure:
- Source
---- Preprocessing.py: contains the PREPROCESS class, which consists of the preprocessing of the datasets characterised in Chapter 2.
---- NODE_CLASS.py and CART_CLASS.py: contain the implementation of the CART algorithm as a base learner on tree-induction.
---- RANDOM_FOREST_CLASS.py and DECISION_FOREST_CLASS.py: implementation of Random Forest and Decision Forest algorithms, respectively, considering CART as the base learner on tree-induction. Even though there are coincident methods, it opted to separate completely the implementations, for further distinguished development purposes.
---- main.py: the main .py file, where the preprocessing, training and testing stages are executed.

- Data
---- hepatitis.arff: small dataset with numerical and, predominantly, categorical data.
---- tic-tac-toe.csv: medium dataset only with categorical data.
---- CureThePrincess.csv: large dataset only with numerical data.
---- accuracy.txt: accuracies from the combinations of hyper-parameters.
---- features_relevance.txt: feature list relevance for each combination.

- Outputs
---- accuracy.txt: accuracies from the combinations of hyper-parameters.
---- features_relevance.txt: feature list relevance for each combination.

- README.txt