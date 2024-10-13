import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 2, 1.3, 3.75, 2.25])

X_mean = np.mean(X)
Y_mean = np.mean(Y)

numerator = np.sum((X - X_mean) * (Y - Y_mean))
denominator = np.sum((X - X_mean) ** 2)
b1 = numerator / denominator
b0 = Y_mean - (b1 * X_mean)

Y_pred = b0 + b1 * X

plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Regression Line')
plt.xlabel('X (Independent Variable)')
plt.ylabel('Y (Dependent Variable)')
plt.legend()
plt.show()

print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
