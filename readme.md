The program starts by running main function in main.py.

By default, the program runs on "CM1.arff" located in NASA_MDP/MDP/D' folder, using all 6 models defined in Utilities.py
The result is saved in the src folder with respective names.

Many parameters can be tuned inside Utilities.py, however, some params changed may result in breaking some part of code.

Every algorithm is initiated by their respective functions in main.py.
I.e. start_regular_algorithm() for training without Feature Selection, start_genetic_algorithm() for training with Feature Selection, etc.