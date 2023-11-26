Supervised and Experiential Learning
Practical Work 3: a CBR prototype for a synthetic task: planning / design / configuration in a concrete domain
Joao Agostinho Valerio
Eirik Armann Grytøyr
Clara Rivadulla Duró
John Kelly Villota Pismag
15/06/2023

Requirements:
customtkinter==5.1.3
ipython==8.14.0
numpy==1.24.2
pandas==1.5.3
scikit_learn==1.2.2
xlrd==2.0.1

Software:
PyCharm CE

Programming Language:
Python - version 3.8.8

Execution:
To execute the code through the terminal use the following to ensure the packages are installed:
pip install customtkinter
pip install ipython 
pip install numpy
pip install pandas
pip install scikit_learn
pip install xlrd

To run the Travel planner, double click on the travel_planner.bat (for windows) or travel_planner.sh (for linux)


PW3-SEL-2023-G1 Folder structure:

- Documentation
---- PW3-SEL-2023-G1.pdf: Technical document about the development of the work.
---- PW3-MANUAL-SEL-2023-G1.pdf: User manual.

- Source
---- ui.py: interface shown to the user (this is the file that must be run, when executing the code).
---- main.py: contains the code to test the model with distinct inputs.
---- preprocessing.py: the preprocessing of the data.
---- Retrieve.py: obtain the closest case. 
---- Adaptation.py: adaptation of the closest case.
---- Learning.py: learning process of the model.
---- testing.py: model's tests performed.

- Data
---- attr_info.json: Atrribute description of the data.
---- travel.csv: Case-base dataset.
---- Travel-info.txt: Features' properties description.

- README.txt: structure and contents of the Delivery ZIP file.