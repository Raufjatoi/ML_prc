import numpy as n
from collections import Counter as c

def euclidean_distance(p1, p2):
    return n.sqrt(n.sum((p1 - p2) ** 2))

def knn(Xt, yt, X_test, k=3):
    preds = []
    for tp in X_test:
        dis = [euclidean_distance(tp, x) for x in Xt]
        
        k_indices = n.argsort(dis)[:k]
        
        k_nearest_labels = [yt[i] for i in k_indices]
        
        most_common = c(k_nearest_labels).most_common(1)
        preds.append(most_common[0][0])
    
    return preds


X_train = n.array([
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [5, 6, 7, 8, 9],
    [6, 7, 8, 9, 10],
    [7, 8, 9, 10, 11],
    [10, 11, 12, 13, 14],
    [11, 12, 13, 14, 15],
    [12, 13, 14, 15, 16],
    [14, 15, 16, 17, 18]
])

y_train = n.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])

X_test = n.array([[100, 101, 102, 103, 104]])


predictions = knn(X_train, y_train, X_test, k=3)
print(predictions)
