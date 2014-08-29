import sys
import numpy as np
from operator import add 
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans as km


def KMeansModel(dataPath, label, k, character, master):
    sc = SparkContext(master)
    data = sc.textFile(dataPath).map(lambda line: line.replace(character, ','))

    if label == 0:
        label_sum = data.map(lambda line: line.split(',')).map(lambda data: (float(data[0]), 1)).reduceByKey(add).collect()
        label = data.map(lambda line: line.split(',')).map(lambda data: float(data[0])).collect()        
        train_data = data.map(lambda line: line.split(',')).map(lambda x: map(lambda part: float(part), x[1:len(x)]))
    else:
        label_sum = data.map(lambda line: line.split(',')).map(lambda data: (float(data[-1]), 1)).reduceByKey(add).collect()
        label = data.map(lambda line: line.split(',')).map(lambda data: float(data[-1])).collect()        
        train_data = data.map(lambda line: line.split(',')).map(lambda x: map(lambda part: float(part) if part is not None else '', x[:len(x) - 1]))
    model = km.train(train_data, k)
    predict_data = train_data.collect()
    train = len(predict_data)
    acc = 0
    
    for i in range(len(label_sum)):
        ksum = np.zeros(k, dtype = int)
        cur_label = label_sum[i][0]
        for j in range(train):
            if label[j] == cur_label:
                ksum[model.predict(predict_data[j])] += 1
        acc += max(ksum)

    string = "KMeans Result: \n"
    center = model.centers
    for i in range(k):
        cur = str(i) + ":" + str(center[i]) + '\n'
        string += cur  
    string = string + "Acc: " + str((float(acc)/train) * 100) + "%"    
    sc.stop()
    return string
    #print "test:",train_data.collect()
    #print model.centers
    #return model
        
