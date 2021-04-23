#!/usr/bin/env python3
'''
Created on Apr 23, 2021

@author: lorenzo
'''
import sys
import pandas as pd
import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

class Model: 
    def __init__(self, label_dim):
        self.label_dim = label_dim
        self.data_dim = None
        
    def _init_model(self):
        if self.data_dim is None:
            print("data_dim not set!", file=sys.stderr)
            exit(1)
        
        self.model = tf.keras.models.Sequential()
        # hidden layer
        self.model.add(tf.keras.layers.Dense(int(self.data_dim / 2), activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(int(self.data_dim / 20), activation='relu'))
        # Final output node for prediction 
        self.model.add(tf.keras.layers.Dense(self.label_dim))
        self.model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    def train(self, labels, data):
        self.data_dim = data.shape[1]
        
        # Split
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=1)

        self.scaler = StandardScaler()
        data_train = self.scaler.fit_transform(data_train) 
        data_test = self.scaler.transform(data_test)

        self._init_model()
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        self.model.fit(x=data_train, 
                  y=labels_train, 
                  epochs=250,
                  batch_size=128,
                  validation_data=(data_test, labels_test),
                  verbose=1,
                  callbacks=[early_stop])
        
        losses = pd.DataFrame(self.model.history.history)
        for metric in self.model.metrics_names:
            myplot = losses[['%s' % metric,'val_%s' % metric]].plot()
            myplot.legend(["Train", "Test"])
            myplot.set_xlabel("Epoch", fontsize=20)
            myplot.set_ylabel(metric, fontsize=20)
            myplot.figure.savefig("loss_%s.png" % metric)
            
        return (labels_train, data_train), (labels_test, data_test)
        
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        
    def load_weights(self, data_dim, filepath):
        self.data_dim = data_dim
        self._init_model()
        self.model.load_weights(filepath)
        
    def predict(self, data):
        if data.shape[1] != self.data_dim:
            print("Data's dimension is different from the one used to initialise the model (%d != %d)" % (data.shape[1], self.data_dim), file=sys.stderr)
            exit(1)
            
        return self.model.predict(data, verbose=0).squeeze()

    def print_figures(self, labels, data):
        results = self.predict(data)
        # we join label and results so that we can sort and split the latter according to the former
        if self.label_dim == 1:
            total_test_result = np.transpose(np.vstack((labels, results)))
            total_test_result = total_test_result[labels.argsort()]
            split_total_results = [total_test_result[total_test_result[:,0] == k] for k in np.unique(labels)]
        else:
            total_test_result = np.hstack((labels, results))
            labels = list(np.transpose(x.squeeze()) for x in np.hsplit(labels, 2))
            indexes = np.lexsort(labels)
            total_test_result = total_test_result[indexes]
            split_total_results = [total_test_result[np.all(total_test_result[:,0:self.label_dim] == k, axis=1)] for k in np.unique(total_test_result[:,0:self.label_dim], axis=0)]
        compare_diff = []
        compare_labels = []
        #compare_diff_mode = []
        
        for data in split_total_results:
            compare_label = data[0,0:self.label_dim]
            compare_results = data[:,self.label_dim:]
            
            # hist, bin_edges = np.histogram(compare_results, bins=10, density=True)
            # idx = hist.argmax()
            # bin_length = bin_edges[1] - bin_edges[0]
            # cosmax_range = bin_edges[idx] - bin_length, bin_edges[idx + 1] + bin_length
            # indexes = np.where(np.logical_and(compare_results >= cosmax_range[0], compare_results <= cosmax_range[1]))
            # cosmax_bar = np.average(compare_results[indexes])
            
            compare_labels.append(compare_label)
            compare_diff.append(np.average(compare_results) - compare_label)
            #compare_diff_mode.append(cosmax_bar - compare_label)
            
        compare_labels = np.array(compare_labels)
        compare_diff = np.array(compare_diff)
        #compare_diff_mode = np.array(compare_diff_mode)

        plt.figure(figsize=(12,8))
        _ = plt.scatter(compare_labels, compare_diff / compare_labels)
        #plt.scatter(compare_labels, compare_diff_mode / compare_labels)
        plt.xlabel(r'Label value $l_0$', fontsize=20)
        plt.ylabel(r'Accuracy $(l - l_0) / l_0$', fontsize=20)
        plt.legend(["Average", "Mode"])
        plt.savefig('accuracy.png')

        if self.label_dim == 1:
            n_unique_labels = len(split_total_results)
            print(n_unique_labels, len(np.unique(labels)))
            cols = 6
            rows = int(np.ceil(n_unique_labels / cols))
            fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', figsize=(16,12))
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    if idx < n_unique_labels:
                        data = split_total_results[idx]
                        ax[i, j].set_title("%.2lf" % data[0,0], fontsize=14)
                        ax[i, j].hist(data[:,1], density=True)
                        ax[i, j].axvline(data[0,0], color="red")
            fig.savefig("histograms.png")
            

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage is %s input_file label_dim" % sys.argv[0])
        exit(1)
    
    input_file = sys.argv[1]
    label_dim = int(sys.argv[2])
    
    dataset = np.loadtxt(input_file, delimiter=",")
    # split the dataset into labels and actual data
    labels = dataset[:,0:label_dim].squeeze()
    data = dataset[:,label_dim:]
    
    model = Model(label_dim)
    _, test = model.train(labels, data)
    model.print_figures(test[0], test[1])

