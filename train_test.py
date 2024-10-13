import numpy as n

X = n.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
y = n.array([0, 1, 0, 1, 0, 1, 0, 1])

test_size = 0.5
num_test_samples = int(test_size * len(X))

indices = n.arange(len(X))
n.random.shuffle(indices)

test_indices = indices[:num_test_samples]
train_indices = indices[num_test_samples:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)