from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
np.random.seed(1)

X, y = make_classification(n_samples=10000, n_classes=2, random_state=43)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)

lr = LogisticRegression()
from time import time


parameter_grid = {'C':[0.001,0.01,0.1,1,10], 
                  'penalty':['l1', 'l2']  
                  }

classifier = GridSearchCV(estimator=lr, param_grid=parameter_grid, scoring='accuracy', cv=10)
start = time()
classifier.fit(X_train, y_train)
print(f'Time',time() - start)

best_penalty = classifier.best_params_['penalty']
best_C = classifier.best_params_['C']

print(classifier.best_params_)
print(classifier.best_estimator_)
print(classifier.best_score_)

clf_lr = LogisticRegression(penalty=best_penalty, C=best_C)
clf_lr.fit(X_train, y_train)

predictions = clf_lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Accuracy',accuracy_score(y_test, predictions))