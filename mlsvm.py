import sys
import numpy as np
import matplotlib.pyplot as plt
from normalize import norm
from sklearn.decomposition import PCA
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint as lbp
from pyspark.mllib.classification import SVMWithSGD as svm

def SVMModel(dataPath, label, max_label, min_label, character, master, normalize, ispca):
    
    pca_n = 2
    sc = SparkContext(master)
    data = sc.textFile(dataPath)
    
    mid_label = (float(max_label) + float(min_label)) / 2.0

    print data.map(lambda line: line.split(character)).collect()
 
    ndata = data.map(lambda line: line.split(character)).map(lambda part: (map(lambda x: float(x) ,part[0: len(part)])))

    if label == 0:
        ndata = ndata.map(lambda line: line[::-1])

    if normalize == 1:
        test_data = norm(ndata.collect())    
        norm_data = sc.parallelize(test_data)
        train_data = norm_data.map(lambda part: lbp([1.0 if float(part[0]) > mid_label else 0.0][0], part[1])) 
        test_data = norm_data.map(lambda part: ([1.0 if float(part[0]) > mid_label else 0.0][0], part[1])).collect()

    else:
        train_data = ndata.map(lambda part: lbp([1.0 if float(len(part) - 1) > mid_label else 0.0][0], part[0: len(part) - 1]))
        test_data = ndata.map(lambda part: ([1.0 if float(part[len(part) - 1]) > mid_label else 0.0][0], part[0:len(part) - 1])).collect()

    if ispca == 1:
        pca = PCA(n_components = pca_n)
        pca_train = [test_data[i][1] for i in range(len(test_data))]
        pca_data = pca.fit(pca_train).transform(pca_train)

        test = []
        for i in range(len(pca_data)):
            test.append([test_data[i][0], pca_data[i]])

        train_data = sc.parallelize(test).map(lambda part: lbp(part[0], part[1]))
        test_data = test
    


    model_svm = svm.train(train_data)
    acc_svm = 0
    err_lrg = 0.0
    size = len(train_data.collect())
   
    for i in range(size):
        if model_svm.predict(test_data[i][1]) == test_data[i][0]:
            acc_svm += 1
   
    String = "SVM Result:\n"
    String = String + str(model_svm.weights) + "\n"
    String = String + str((float(acc_svm)/ float(size)) * 100) + "%"
    

    x = []
    y = []
    showpic = 0

    if len(test_data[0][1]) == 2:
        ispca = 1

    if ispca == 1:
        for i in range(size):  
            if test_data[i][0] == 0.0:     
                plt.plot(test_data[i][1][0], test_data[i][1][1], 'ro', color = 'r', markersize = 8)
            elif test_data[i][0] == 1.0:
                plt.plot(test_data[i][1][0], test_data[i][1][1], 'ro', color = 'b', markersize = 8)

        test = sc.parallelize(test_data)
        max_axis = test.map(lambda part: part[1][0]).max()
        min_axis = test.map(lambda part: part[1][0]).min()
        plt.plot([min_axis, max_axis], [max_axis * model_svm.weights[0] + model_svm.weights[1], min_axis * model_svm.weights[0] + model_svm.weights[1]], 'g-', linewidth= 2)
        plt.savefig('result.jpg')
        plt.close('all')
        showpic = 1

    sc.stop()
    return (showpic, String)

