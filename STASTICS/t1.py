import numpy as n
from scipy import stats as s 

scores = n.array([80, 70, 90, 85, 85, 100])

print(f'mean: {n.mean(scores)}')
print(f'median: {n.median(scores)}')
print(f'mode: {s.mode(scores)}')
print(f'variance: {n.var(scores):2f}')
print(f'standard deviation: {n.std(scores):2f}')