import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import pickle


def get_dataset_sizes():
    with open('methedology/onlyNumWords.json') as json_file:
        elar = dict(json.load(json_file))
    with open('methedology/verified_dict_spottings.json') as json_file:
        bobsl_verified = dict(json.load(json_file))
    with open('methedology/dict_spottings.json') as json_file:
        bobsl_unverified = dict(json.load(json_file))
    return elar, bobsl_verified, bobsl_unverified


def plot_bar_sizes(bobsl_unverified_size, bobsl_verified_size, elar_size):
    datasets = ['BOBSL unverified', 'BOBSL verified', 'ELAR']
    size = [bobsl_unverified_size, bobsl_verified_size, elar_size]

    for i in range(len(datasets)):
        plt.text(datasets[i], size[i], str(size[i]), ha='center', va='bottom')


    plt.bar(datasets, size)
    plt.xlabel('Datasets')
    plt.ylabel('Numer of unique words')
    plt.title('Sizes of the dataset')
    plt.show()

def get_splits():
    # Save Data
    X_train = np.loadtxt("methedology/X_train.npy")
    X_test = np.loadtxt("methedology/X_test.npy")
    y_train = np.loadtxt("methedology/y_train.npy")
    y_test = np.loadtxt("methedology/y_test.npy")
    
    return X_train, X_test, y_train, y_test

def get_models():
    with open('methedology/KNNdisp.pkl', 'rb') as file:
        knn: DecisionBoundaryDisplay = pickle.load(file) 

    with open('methedology/DTCdisp.pkl', 'rb') as file:
        dtc: DecisionBoundaryDisplay = pickle.load(file) 

    with open('methedology/RFCdisp.pkl', 'rb') as file:
        rfc: DecisionBoundaryDisplay = pickle.load(file) 
    return knn, dtc, rfc

def decision_boundaries(disp, X_train, y_train):
    disp.ax_.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k")

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

def plot_bar(labels, values, xlabel, ylabel, title):
    # for i in range(len(labels)):
    #     plt.text(labels[i], values[i], str(values[i]), ha='center', va='bottom')

    plt.figure(figsize=(20, 6))
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

