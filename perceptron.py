import glob
import json
import locations
import numpy as np
import os
import pickle
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import viti

layers = ['cnn',1,6,12,18,21,24]
sections = ['vowel', 'syllable', 'word']



def _get_layer(filename, layer = 'logits'):
    outputs = viti.load_outputs(filename)
    if layer == 'logits':
        return outputs.logits.detach().numpy()
    layer_index = int(layer)
    if layer_index > len(outputs.hidden_states):
        raise Exception('layer index too high')
    return outputs.hidden_states[layer_index].detach().numpy()

def load_stamp_data(layer = 'logits'):
    directory = locations.viti_outputs 
    stamp = glob.glob(directory + 'with_stamp*.pickle')
    no_stamp = glob.glob(directory + 'no_*.pickle')
    no_stamp += glob.glob(directory + 'with_note*.pickle')
    y = [1] * len(stamp)
    y += [0] * len(no_stamp)
    X = np.zeros([len(y), 1000])
    for index, filename in enumerate(stamp + no_stamp):
        X[index] = _get_layer(filename, layer)
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        stratify=y,random_state=1)
    return X_train, X_test, y_train, y_test
    
def train_stamp_classifier(layer = 'logits', use_all_data = False):
    X, y = load_stamp_data()
    if use_all_data: 
        X_train, X_test, y_train, y_test = X, X, y, y
    else: 
        X_train, X_test, y_train, y_test = split_data(X, y)
    clf=MLPClassifier(hidden_layer_sizes=(300,),random_state=1,max_iter=500)
    clf.fit(X_train, y_train)
    hyp = clf.predict(X_test)
    gt = y_test
    return clf, gt, hyp 

def accuracy(gt, hyp):
    return accuracy_score(gt, hyp)

def mcc(gt, hyp):
    return matthews_corrcoef(gt, hyp)

def report(gt, hyp):
    print(classification_report(gt, hyp))


def save_classifier(clf, filename = 'stamp_classifier.pickle'):
    pickle.dump(clf, open(filename, 'wb'))

def load_classifier(filename = 'stamp_classifier.pickle'):
    clf = pickle.load(open(filename, 'rb'))
    return clf
