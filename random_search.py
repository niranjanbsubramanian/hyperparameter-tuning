from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
np.random.seed(1)

X, y = make_classification(n_samples=10000, n_classes=2, random_state=43)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)

lr = LogisticRegression()
from time import time


parameter_grid = {'C':np.logspace(-2,1,100), 
                  'penalty':['l1', 'l2']  
                  }

random_search = RandomizedSearchCV(estimator=lr, param_distributions=parameter_grid, n_iter=7, scoring='accuracy', cv=10,n_jobs=-1)
start = time()
random_search.fit(X_train, y_train)
print(f'Time',time() - start)

print(random_search.best_params_)
print(round(random_search.best_score_, 3))

#np.logspace(-3,2,num=2-(-3), base=10)

best_penalty = random_search.best_params_['penalty']
best_C = random_search.best_params_['C']

clf_lr = LogisticRegression(penalty=best_penalty, C=best_C)
clf_lr.fit(X_train, y_train)

predictions = clf_lr.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

print(f'Best Penalty:', best_penalty)
print(f'Best C:', best_C)