import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import pandas as pd

# LOAD DATA FUNCTIONS------------------------------------------------------------------------------------------------

def get_path_movement():
    pathMovement = {}  
    f = open("keyBinds.txt", "r")
    line = f.readline()
    i= 0
    j = 0
    while (line):
        try:
            if line.split("_|_")[1] == 'None\n':
                pathMovement[line.split("_|_")[0].upper()] = None
            else:
                pathMovement[line.split("_|_")[0].upper()] = line.split("_|_")[1]
        except:
            i += 1
        line = f.readline()
    f.close()
    return pathMovement

def get_primary_location():
    Locations = {}
    f = open("keyBindsLocation.txt", "r")
    line = f.readline()
    while (line):
        try:
            if line.split("_|_")[1] == 'None\n':
                Locations[line.split("_|_")[0]] = None
            else:
                Locations[line.split("_|_")[0]] = line.split("_|_")[1]
        except:
            i = 1
        line = f.readline()
    f.close()
    return Locations


def get_usability():
    usability = {}
    f = open("useability.txt", "r")
    line = f.readline()
    while (line):
        try:
            if line.split("_|_")[1] == 'None\n':
                usability[line.split("_|_")[0]] = None
            else:
                usability[line.split("_|_")[0]] = int(line.split("_|_")[1])
        except:
            if line != '\n':
                print(line)
        line = f.readline()
    f.close()
    return usability

def count_usability(usability):
    numUsablility = {0:0,1:0,2:0,3:0,4:0}
    for key, value in usability.items():
        numUsablility[value] += 1
    return numUsablility

def get_primary_movement_keybinds():
    label_keys = {"Up\n":0, "Down\n":1, "Up and Down\n":2, "Sideways\n":3, 
              "Side to Side\n":4, "Away\n":5, "Towards\n":6, 
              "Back and Forth\n":7, "Horizontal Circular\n":8, 
              "Vertical Circular\n":9, "Local Movement\n":10, "Unknown\n": 11,}
    return label_keys

def get_primary_location_keybinds():
    label_keys = {"Head\n":0, "Eye\n":1, "Forehead\n":2, "Nose\n":3, 
              "Ear\n":4, "Mouth\n":5, "Chin\n":6, 
              "Neck\n":7, "Shoulders\n":8, 
              "ArmPit\n":9, "Chest\n":10, "Waist\n": 11,
              "Back\n":12, "Thigh\n":13, "Stomach\n": 14,
              "Arm\n":15, "Wrist\n":16, "Hand\n": 17,
              "Neutral\n": 18}
    return label_keys


def load_imputed_Data(coorDataLocation="D:/Thesis/ELAR-Data/imputedArrayData/"):
    X = []
    title = []
    dirData = os.listdir(coorDataLocation)
    for i, data in enumerate(dirData):
        if i % 2:
            continue
        coorLoad = np.loadtxt(coorDataLocation + data)
        coorShape= np.loadtxt(coorDataLocation + dirData[i + 1])
        
        try:
            coorLoad = coorLoad.reshape(coorLoad.shape[0], coorLoad.shape[1] // int(coorShape[2]), int(coorShape[2]))
        except:
            print(data, dirData[i + 1], coorLoad.shape, coorShape.shape)
        title.append(data)
        X.append(coorLoad)

    newTitle = []
    for name in title:
        newName = name[:-4] + '.mp4'
        newTitle.append(newName)
    title = np.array(newTitle)
    
    return X, title

def assign_classification_movement(title, pathMovement, label_keys):
    videoWord = []
    for i, data in enumerate(title):
        data = data.upper()
        movement = pathMovement[data.split("_")[0]]
        if movement is None:
            videoWord.append(99)
        else:
            videoWord.append(label_keys[movement])
    return videoWord

def create_x_y_movement(X, videoWord, usability, title):
    newX = []
    newTitle = []
    Y = []
    # Append X values with Y dataLabels
    for i, y in enumerate(videoWord):
        # 99 was assigned if an error occured when importing and 11 is for unknown datalabels
        if y != 99 and y != 11:
            newX.append(X[i])
            newTitle.append(title[i])
            Y.append(y)
    X = newX
    title = np.array(newTitle)
    # Remove X and Y that are low usability
    newX = []
    newY = []
    for i in range(len(X)):
        if usability[title[i]] >= 3:
            newX.append(X[i])
            newY.append(Y[i])
    X = np.array(newX)
    Y = np.array(newY)
    # check shape of data
    newX = []
    newY = []
    for i in range(len(X)):
        try:
            if X[i].shape[1] == 33 or X[i].shape[2] == 3:
                newX.append(X[i])
                newY.append(Y[i])
        except:
            continue
    X = np.array(newX)
    Y = np.array(newY)
    return X, Y

def squeeze_one_dimension(X):
    newX = []
    for i in range(len(X)):
        newInfo = []
        for frame in X[i]:
            for joint in frame:
                for coor in joint:
                    newInfo.append(coor)
        newX.append(newInfo)
    X = np.array(newX)
    return X

def get_splits(experiment=1):
    experimentData = ['experiment1', 'experiment2', 'experiment3', 'experiment4', 'experiment5', 'experiment6', 'experiment7', 'experiment8']
    folder = experimentData[experiment - 1]
    # Save Data
    X_train = np.loadtxt(folder + "/X_train.npy")
    X_test = np.loadtxt(folder + "/X_test.npy")
    y_train = np.loadtxt(folder + "/y_train.npy")
    y_test = np.loadtxt(folder + "/y_test.npy")
    
    return X_train, X_test, y_train, y_test

def get_X_Y(X_train, X_test, y_train, y_test):
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((y_test, y_train))
    return X, Y


def data_pandas(y, label_keys, remove):
    unique_classes, counts = np.unique(y, return_counts=True)
    invs_label_key = {label_keys[key]: key for key in label_keys.keys()}
    class_counts_df = pd.DataFrame({'Class': [invs_label_key[i][:-remove] for i in unique_classes], 'Count': counts})
    return class_counts_df

def get_stratified_splits(X, Y):
    # Random state
    seed_value = 42  
    random.seed(seed_value)

    # Choose the test size
    pick = False

    yCount = {}
    if pick:
        testSize = 0.33
    else:
        for value in Y:
            if value in yCount:
                yCount[value] += 1
            else:
                yCount[value] = 1
        trainSize = min(yCount.values()) * len(yCount.keys())
        testSize =  1 -(trainSize / len(Y))

    # set up random picking
    picked = {i:0 for i in range(len(X))}
    Yset = set(Y)
    uniqueYs = len(Yset)
    Ordered_X_train, X_test, Ordered_y_train, y_test = ([],[],[],[])

    # Randomly add and even amoung of training data based off test size
    numAdded = 1
    while ((len(Ordered_X_train)/ len(X)) <  (1 - testSize)):
        for y in range(uniqueYs):
            index = np.random.randint(0, len(Y) - 1)
            while picked[index] == 1 or not Y[index] == y:
                index = random.randint(0, len(Y) - 1)
                if yCount[y] <= numAdded:
                    break
            picked[index] = 1
            Ordered_X_train.append(X[index])
            Ordered_y_train.append(y)
        numAdded += 1
        
    # Add remaing values into test
    for i, bool in picked.items():
        if not bool:
            X_test.append(X[i])
            y_test.append(Y[i])
    print("Test set created")

    # Randomly order training:
    X_train, y_train = ([],[])
    pickedReorder = {i:0 for i in range(len(Ordered_y_train))}
    print(len(pickedReorder.values()))
    index = np.random.randint(0, len(Ordered_y_train) - 1)
    num = 0
    while(np.prod(list(pickedReorder.values())) == 0):
        while pickedReorder[index] == 1:
            index = random.randint(0, len(Ordered_y_train) - 1)
        pickedReorder[index] = 1
        X_train.append(Ordered_X_train[index])
        y_train.append(Ordered_y_train[index])
        num += 1
    print("Training set created")
    return X_train, X_test, y_train, y_test

# finish this later
def remove_local_movement(X, Y):
    newX = []
    newY = []
    for i in range(len(Y)):
        if Y[i] != 11:
            newX.append(X[i])
            newY.append(Y[i])


# EXPERIMENT HELPER FUNCTIONS ---------------------------------------------------------------------------------------------
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import f1_score

def result_data_frame(modelNames, trainScore, testScore):
    return pd.DataFrame(np.array([modelNames, trainScore, testScore]).T, columns=['Model', 'Training Accuracy', 'Test Accuracy'])

def result_data_frame_no_name(modelNames, trainScore, testScore):
    return pd.DataFrame(np.array([trainScore, testScore]).T)


def print_data_Frame(df):
    print(df.to_markdown())

def plot_confusion(label_keys, model, X_test, y_test, model_name, num_labels):
    inverted_label_keys = dict(map(reversed, label_keys.items()))
    label = [inverted_label_keys[i] for i in range(num_labels)]

    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_pred, y_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title(model_name)
    plt.show()

def get_f1score(model, X_test, y_test,):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1

def plot_confusion_location(label_keys, model, X_test, y_test, model_name, num_labels):
    inverted_label_keys = dict(map(reversed, label_keys.items()))
    label = [inverted_label_keys[i] for i in range(num_labels)]

    y_pred = model.predict(X_test)
    y_true = y_test.copy()
    numberLabel = np.arange(num_labels)

    if not set(numberLabel).issubset(set(y_test)):
        missing_class = list(set(numberLabel) - set(y_test))
        y_true = np.concatenate((y_true, missing_class))
        y_pred = np.concatenate((y_pred, missing_class))

    cm = metrics.confusion_matrix(y_pred, y_true)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title(model_name)
    plt.show()


def class_accuracy(model, X_test, y_test, label_list):
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_pred, y_test)
    accuracies =  cm.diagonal() / cm.sum(axis=1)
    return pd.DataFrame(np.array([label_list, accuracies]).T)

def class_accuracy_loaction(model, X_test, y_test, label_list, num_labels):
    y_pred = model.predict(X_test)
    y_true = y_test.copy()
    numberLabel = np.arange(num_labels)
    if not set(numberLabel).issubset(set(y_test)):
        missing_class = list(set(numberLabel) - set(y_test))
        y_true = np.concatenate((y_true, missing_class))
        y_pred = np.concatenate((y_pred, missing_class))
    cm = metrics.confusion_matrix(y_pred, y_true)
    diagonalList = []
    for diagonal in  cm.diagonal():
        if diagonal == 1:
            diagonalList.append(0)
        else:
            diagonalList.append(diagonal)
    diagonalList = np.array(diagonalList)
    accuracies =  diagonalList / cm.sum(axis=1)
    return pd.DataFrame(np.array([label_list, accuracies]).T)

# LOAD EXPERIMENT FUNCTION ------------------------------------------------------------------------------------------------
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_experiement(experiment=1):
    experimentData = ['experiment1', 'experiment2', 'experiment3', 'experiment4', 'experiment5']
    experimentFolder = experimentData[experiment - 1]
    with open(experimentFolder + '/LDA.pkl', 'rb') as file:
        lda: LinearDiscriminantAnalysis = pickle.load(file)
    
    with open(experimentFolder + '/QDA.pkl', 'rb') as file:
        qda: QuadraticDiscriminantAnalysis = pickle.load(file) 

    with open(experimentFolder + '/GNB.pkl', 'rb') as file:
        gnb: GaussianNB = pickle.load(file) 

    with open(experimentFolder + '/KNN.pkl', 'rb') as file:
        knn: KNeighborsClassifier = pickle.load(file) 

    with open(experimentFolder + '/DTC.pkl', 'rb') as file:
        dtc: DecisionTreeClassifier = pickle.load(file) 

    with open(experimentFolder + '/RFC.pkl', 'rb') as file:
        rfc: RandomForestClassifier = pickle.load(file) 

    return lda, qda, gnb, knn, dtc, rfc
