import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

train = np.array(pd.read_csv("human-activity-recognition-with-smartphones/train.csv",low_memory=False))
train_labels = train.T[len(train.T)-1].T
train_features = train.T[0: len(train.T)-3].T

test = np.array(pd.read_csv("human-activity-recognition-with-smartphones/test.csv",low_memory=False))
test_labels = test.T[len(test.T)-1].T
test_features = test.T[0: len(test.T)-3].T

print(train_features.shape)
print(test_features.shape)

all_features = np.concatenate((train_features,test_features))
all_labels = np.concatenate((train_labels,test_labels))

pca = PCA(n_components = 0.99)
principleComp = pca.fit_transform(all_features)

print(pd.DataFrame(data=principleComp))

kf = KFold(n_splits=10)

results = []
accuracies = []
for train_index, test_index in kf.split(principleComp):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = principleComp[train_index], principleComp[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

    svc = SVC(gamma="auto", decision_function_shape='ovr')
    svc.fit(x_train,y_train)

    predict = svc.predict(x_test)

    print(confusion_matrix(y_test,predict))
    print(classification_report(y_test,predict))
    print(accuracy_score(y_test,predict))
    
    results.append(list([x_train,y_train,x_test,y_test]))
    accuracies.append(accuracy_score(y_test,predict))

print("Best Accuracy",max(accuracies))
print("Best Accuracy index",accuracies.index(max(accuracies)))

"""parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 10]}

svm = GridSearchCV(SVC(gamma="auto"),parameters,cv=10)
svm.fit(principleComp,all_labels)
print("Best Score: ",svm.best_score_)
print("Best Index: ",svm.best_index_)
print("Best Estimator: ",svm.best_estimator_)
print("Best Params: ",svm.best_params_)"""