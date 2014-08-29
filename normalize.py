import numpy as np

def norm(data):

    max_data = np.zeros(len(data[0]) - 1)
    min_data = np.array([10000.0] * (len(data[0]) - 1))

    for i in range(len(data)):
        for j in range(len(data[0]) - 1):

            if data[i][j] > max_data[j]:
                max_data[j] = data[i][j]

            elif data[i][j] < min_data[j]:
                min_data[j] = data[i][j]

    norm_data = []

    for i in range(len(data)):
        norm_data.append((data[i][len(data[0]) - 1],((np.array(data[i][0:len(data[0]) - 1]) - np.array(min_data)) / (np.array(max_data) - np.array(min_data)))))
    
    return norm_data