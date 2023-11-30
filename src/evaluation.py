import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve,precision_recall_curve
from sklearn.metrics import precision_recall_curve,f1_score, roc_auc_score

df_train = pd.read_csv('../data/train.csv')
df_train

X = df_train.drop('booking_status', axis=1)
y = df_train['booking_status']

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

valor = y_resampled.value_counts()
suma = valor.sum()
porcentaje = valor*100/suma
print(porcentaje)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

print(X_resampled.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)


pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("selectkbest", SelectKBest()),
    ("classifier", RandomForestClassifier())
])

log_params = {
    'selectkbest__k':np.arange(10,20),
    'classifier': [LogisticRegression()],
    'classifier__C': [0.1,1,10]
}
rf_params = {
    'scaler': [StandardScaler(), None],
    'selectkbest__k':np.arange(10,20),
    'classifier': [RandomForestClassifier()],
    'classifier__max_depth': [7,9,11]
}
gb_params = {
    'scaler': [StandardScaler(), None],
    'selectkbest__k':np.arange(10,20),
    'classifier': [GradientBoostingClassifier()],
    'classifier__max_depth': [7,9,11]
}
knn_params = {
    'selectkbest__k':np.arange(10,20),
    'classifier': [KNeighborsClassifier()],
    'classifier__n_neighbors': np.arange(10,20)
}
svm_params = {
    'selectkbest__k':np.arange(10,20),
    'classifier': [SVC()],
    'classifier__C': [0.1,1,10]
}

search_space = [
    log_params,
    rf_params,
    gb_params,
    knn_params,
    svm_params   
]

clf_gs = GridSearchCV(estimator=pipe, param_grid=search_space, cv=5, scoring="accuracy", verbose=3, n_jobs=-1)

clf_gs.fit(X_train, y_train)

print(clf_gs.best_estimator_)
print(clf_gs.best_score_)
print(clf_gs.best_params_)

final_model = clf_gs.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("accuracy_score", accuracy_score(y_test, y_pred))
print("precision_score", precision_score(y_test, y_pred))
print("recall_score", recall_score(y_test, y_pred))
print("roc_auc_score", roc_auc_score(y_test, y_pred))
print("confusion_matrix\n", confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred, normalize='true'), annot=True)

import pickle

filename = 'finished_model.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(final_model, archivo_salida)