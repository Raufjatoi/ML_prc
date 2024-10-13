import numpy as n

class SVM:
    def __init__(self, lr=0.001, epochs=1000, lambda_param=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = n.where(y <= 0, -1, 1)  
        self.w = n.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (n.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - n.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = n.dot(X, self.w) - self.b
        return n.sign(linear_output)

X = n.array([[1, 2], [2, 3], [3, 4], [1, 5], [2, 1]])
y = n.array([1, 1, -1, -1, 1])
svm = SVM()
svm.fit(X, y)
predictions = svm.predict(X)
print(predictions)
