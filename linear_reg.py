# y = mX + b 

import numpy as n

# data 
X = n.array([1,2,3,4,5])
y = n.array([5,7,9,11,13])

m = 0 # m is slope(coefficient)
b = 0 # b is bias 
# X and y are yk X(inputs or features) and y(output or target)
lr = 0.01 # learning rate 
epochs = 1000 # times to repeat on 
n = len(X) #num of datasets 

# gd algo 
for _ in range(epochs):
    y_pred = m * X + b # lr formula on the top 
    D_m = (-2/n) * sum(X * (y-y_pred)) # der of m 
    D_b = (-2/n) * sum(y-y_pred) #der of b
    m = m - lr * D_m # update m 
    b = b - lr * D_b # update b

print(f'slope m : {m}')
print(f'intercept b : {b}')

y_pred = m * X + b 
print('predicted vals :', y_pred)
print('actual vals : ', y)
print(f'loss : {y - y_pred}')
