from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)

print(f'accuracy: {accuracy_score(y_test , model.predict(X_test)):2f}')
print(f'Precision: {precision_score(y_test, model.predict(X_test), average="macro"):1f}')
print(f'Recall: {recall_score(y_test, model.predict(X_test), average="macro"):1f}')
print(f'F1 Score: {f1_score(y_test, model.predict(X_test), average="macro"):1f}')