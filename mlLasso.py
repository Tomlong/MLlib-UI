import sys
import numpy as np
from normalize import norm
from sklearn.decomposition import PCA
from pyspark import SparkContext
from pyspark.mllib.regression import LassoWithSGD as larg
from pyspark.mllib.regression import LabeledPoint as lbp


def LassoModel(dataPath, label, normalize, character, master, ispca):

    pca_n = 2
    sc = SparkContext(master)
    data = sc.textFile(dataPath)

# not RDD data 

    ndata = data.map(lambda line: line.split(character)).map(lambda part: (map(lambda x: float(x) ,part[0: len(part)])))

    if label == 0:
        ndata = ndata.map(lambda line: line[::-1])

    if normalize == 1:
        test_data = norm(ndata.collect())    
        norm_data = sc.parallelize(test_data)
        train_data = norm_data.map(lambda part: lbp(part[0], part[1]))   
    

    else:
        test_data = ndata.map(lambda part: (part[0], part[1:len(part) - 1])).collect()
        train_data = ndata.map(lambda part: lbp(part[0], part[1: len(part) - 1]))

    if ispca == 1:
        pca = PCA(n_components = pca_n)
        pca_train = [test_data[i][1] for i in range(len(test_data))]
        pca_data = pca.fit(pca_train).transform(pca_train)

        test = []
        for i in range(len(pca_data)):
            test.append([test_data[i][0], pca_data[i]])

        train_data = sc.parallelize(test).map(lambda part: lbp(part[0], part[1]))
        test_data = test
    
    model_larg = larg.train(train_data)
    err_larg = 0.0
    size = len(train_data.collect())

    
    for i in range(size):
        err_larg = err_larg + abs(model_larg.predict(test_data[i][1]) - test_data[i][0]) 
    
    print "result:", err_larg/size

    String = "Lasso Regression Result:\n"
    String = String + str(model_larg.weights) + '\n'
    String = String + "Error: " + str(err_larg / size) 

    sc.stop()

    return String

