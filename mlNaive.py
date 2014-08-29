import sys
import numpy as np
from normalize import norm
from sklearn.decomposition import PCA
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint as lbp
from pyspark.mllib.classification import NaiveBayes as nb

def NaiveBayesModel(dataPath, label, character, master, normalize, ispca):
    
    pca_n = 2
    sc = SparkContext(master)
    data = sc.textFile(dataPath)
    
    ndata = data.map(lambda line: line.split(character)).map(lambda part: (map(lambda x: float(x) ,part[0: len(part)])))

    if label == 0:
        ndata = ndata.map(lambda line: line[::-1])

    if normalize == 1:
        test_data = norm(ndata.collect())    
        norm_data = sc.parallelize(test_data)
        train_data = norm_data.map(lambda part: lbp(part[0], part[1])) 
        test_data = norm_data.map(lambda part: (part[0], part[1])).collect()

    else:
        test_data = ndata.map(lambda part: (part[len(part) - 1], part[0:len(part) - 1])).collect()
        train_data = ndata.map(lambda part: lbp(part[len(part) - 1], part[0: len(part) - 1]))

    if ispca == 1:
        pca = PCA(n_components = pca_n)
        pca_train = [test_data[i][1] for i in range(len(test_data))]
        pca_data = pca.fit(pca_train).transform(pca_train)

        test = []
        for i in range(len(pca_data)):
            test.append([test_data[i][0], pca_data[i]])

        train_data = sc.parallelize(test).map(lambda part: lbp(part[0], part[1]))
        test_data = test               

    model_nb = nb.train(train_data)
    acc_nb = 0
    err_nbg = 0.0
    size = len(train_data.collect())

    for i in range(size):
        if model_nb.predict(test_data[i][1]) == test_data[i][0]:
            acc_nb += 1
  
    String = "NaiveBayes Result:\n"
    String = String + str(model_nb.labels) + "\n"
    String = String + str((float(acc_nb)/ float(size)) * 100) + "%"
    sc.stop()
    return String

