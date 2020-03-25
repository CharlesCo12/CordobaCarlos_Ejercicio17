import pandas as pd
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
import numpy as np
import matplotlib.pylab as plt

data=pd.read_csv('datos.arff')
lineas=set(np.where(np.array(data) =='?')[0])
dat=data.copy()
dat_f=dat.drop(lineas, 0)
# Crea un nuevo array que sera el target, 0 si quebró, 1 si no quebró
quiebra = np.ones(len(dat_f), dtype=int)
ii = np.array(dat_f['Quiebra']==0)
quiebra[ii] = 0

dat_f['Target'] = quiebra

# Crea un dataframe con los predictores
predictors = list(dat_f.keys())
predictors.remove('Target')
predictors.remove('Quiebra')

x_train, x_prima, y_train, y_prima = sklearn.model_selection.train_test_split(dat_f[predictors], dat_f['Target'], test_size=0.5)
x_test, x_f, y_test, y_f = sklearn.model_selection.train_test_split(x_prima, y_prima, test_size=0.6)
n_trees = np.arange(1,400,25)
f1_train = []
f1_test = []

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))

index=np.where(f1_test==np.max(f1_test))[0][0]
n_max=n_trees[index]
plt.scatter(n_trees, f1_test)
plt.scatter(n_trees[index],f1_test[index],c='red')
plt.title('Número Máximo = {} y el valor del F1 Máximo es de: {:.2f}'.format(n_max,np.max(f1_test)))
plt.xlabel('Número de árboles')
plt.ylabel('F1_Score')
plt.savefig('f1.png')
plt.close()

plt.figure(figsize=(30,8))
clf_maximo = sklearn.ensemble.RandomForestClassifier(n_estimators=n_max, max_features='sqrt')
clf_maximo.fit(x_train, y_train)
avg_importance = clf_maximo.feature_importances_
a = pd.Series(avg_importance, index=predictors)
a.nlargest().plot(kind='barh')
plt.xlabel('Feature Importance')
m=sklearn.metrics.f1_score(y_f, clf_maximo.predict(x_f))
plt.title('Número Máximo = {} y el valor del F1 para la validacion es de: {:.2f}'.format(n_max,m))
plt.savefig('features.png')
plt.close()