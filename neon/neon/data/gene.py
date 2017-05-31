import pandas as pd
from sklearn import model_selection
import numpy as np
def load_data(filepath, filename, names, validation_size):

    filepath = filepath + '/' + filename

    # Reading CSV file
    dataset = pd.read_csv(filepath, names=names)

    arr = dataset.values

    # Extracting Characters of DNA sequence
    data = arr[:,1:8]

    # Extracting label (or) class type EI/IF/N
    labels = arr[:,0]
    label = []

    # Based on class type, converting str labesl to int lables 
    for cl in labels:
        if cl == str('ei'):
            label.append(int(0))
        elif cl == str('ie'):
            label.append(int(1))
        elif cl == str('n'):
            label.append(int(2))
    seed = 7

    # Extracing 80% of 3190 splice dataset values for training i.e 2552
    # Getting 20% of 3190 splice dataset values for validation i.e 638
    train_data = np.zeros((2552, 4))
    valid_data = np.zeros((638, 4))
    train_label = np.zeros((2552,1))
    valid_label = np.zeros((638, 1))
   
    # Using sklearn implemenation to split the train and validation datasets
    train_data, valid_data, train_label, valid_label = model_selection.train_test_split(data, label, test_size=validation_size, random_state=seed)

    t_label = np.zeros((2552, 1))
    v_label = np.zeros((638, 1))

    for i, l in enumerate(train_label):
        t_label[i] = int(l)

    for i, l in enumerate(valid_label):
        v_label[i] = int(l)

    return train_data, valid_data, t_label, v_label


