import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import time
import matplotlib.pyplot as plt

def svmApplier(trainSet,trainLabel,gammaFunc,krnl,datasetType,c):
    results = []
    accuracies = []
    training_times = []
    count = 1
    kf = KFold(n_splits=10)
    if(c=="Not defined"):
        svc = SVC(gamma=gammaFunc, decision_function_shape='ovr',kernel=krnl)
    else:
        svc = SVC(gamma=gammaFunc, decision_function_shape='ovr',kernel=krnl,C=c)
    for train_index, test_index in kf.split(trainSet):
        
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = trainSet[train_index], trainSet[test_index]
        y_train, y_test = trainLabel[train_index], trainLabel[test_index]
        if(datasetType == "unbalanced"):

            x_train = np.concatenate((x_train[y_train!="LAYING"],x_train[y_train=="LAYING"][0:5]))
            y_train = np.concatenate((y_train[y_train!="LAYING"],y_train[y_train=="LAYING"][0:5]))

        start_time = time.time()      
        svc.fit(x_train,y_train)
        training_times.append(time.time() - start_time)

        predict = svc.predict(x_test)

        print(confusion_matrix(y_test,predict))
        print(classification_report(y_test,predict))
        print(accuracy_score(y_test,predict))
        print("Fold: ",count, " accuracy:", accuracy_score(y_test,predict) )
        results.append(list([x_train,y_train,x_test,y_test]))
        accuracies.append(accuracy_score(y_test,predict))
        print()
        count += 1

    #print("Accuracies Accuracy SVM",accuracies)
    print("Average Accuracy SVM",np.average(accuracies))
    #print("Best Accuracy index",accuracies.index(max(accuracies)))
    return np.average(accuracies),np.average(training_times)


def kFold_SVM(principleComp,trainLabel,datasetType):
    avg_accuracies = []
    avg_times = []

    print("\nKernel = Linear\n")
    avg_acc,avg_time = svmApplier(principleComp,trainLabel,"auto","linear",datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"linear"))
    avg_times.append(avg_time)

    print("\nKernel = poly\n")
    avg_acc,avg_time = svmApplier(principleComp,trainLabel,"auto","poly",datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"poly"))
    avg_times.append(avg_time)

    print("\nKernel = rbf\n")
    avg_acc,avg_time = svmApplier(principleComp,trainLabel,"auto","rbf",datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"rbf"))
    avg_times.append(avg_time)

    print("\nKernel = sigmoid\n")
    avg_acc,avg_time = svmApplier(principleComp,trainLabel,"auto","sigmoid",datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"sigmoid"))
    avg_times.append(avg_time)

    accuracy =list()
    for i in range(len(avg_accuracies)):
        accuracy.append(avg_accuracies[i][0])

    plt.figure(1)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(["linear","poly","rbf","sigmoid"],avg_times,"-o")
    plt.xlabel("Kernel types")
    plt.ylabel("Average elapsed times in seconds")
    plt.title("Average elapsed time vs kernel type")

    plt.subplot(1,2,2)
    plt.bar(["linear","poly","rbf","sigmoid"],accuracy)
    plt.xlabel("Kernel types")
    plt.ylabel("Average accuracy")
    plt.title("Average accuracy vs gamma value")
    plt.show()

    avg_accuracies = sorted(avg_accuracies,reverse=True)
    optimal_kernel_type = avg_accuracies[0][1]
    
    avg_accuracies = []
    avg_times = []

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,0.001,optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"0.001"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,0.01,optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"0.01"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,0.1,optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"0.1"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,1,optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"1"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,10,optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"10"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,float("inf"),optimal_kernel_type,datasetType,"Not defined")
    avg_accuracies.append((avg_acc,"inf"))
    avg_times.append(avg_time)

    accuracy =list()
    for i in range(len(avg_accuracies)):
        accuracy.append(avg_accuracies[i][0])

    plt.figure(2)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(["0.001","0.01","0.1","1","10","inf"],avg_times,"-o")
    plt.xlabel("Gamma value")
    plt.ylabel("Average elapsed times in seconds")
    plt.title("Average elapsed time vs gamma value")
    plt.subplot(1,2,2)
    plt.bar(["0.001","0.01","0.1","1","10","inf"],accuracy)
    plt.xlabel("Gamma value")
    plt.ylabel("Average accuracy")
    plt.title("Average accuracy vs gamma value")
    plt.show()

    optimal_gamma_value = accuracy[np.argmin(avg_times)]

    avg_accuracies = []
    avg_times = []

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,0.0001)
    avg_accuracies.append((avg_acc,"0.0001"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,0.001)
    avg_accuracies.append((avg_acc,"0.001"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,0.01)
    avg_accuracies.append((avg_acc,"0.01"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,0.1)
    avg_accuracies.append((avg_acc,"0.1"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,1)
    avg_accuracies.append((avg_acc,"1"))
    avg_times.append(avg_time)

    avg_acc,avg_time = svmApplier(principleComp,trainLabel,optimal_gamma_value,optimal_kernel_type,datasetType,float("inf"))
    avg_accuracies.append((avg_acc,"inf"))
    avg_times.append(avg_time)

    accuracy =list()
    for i in range(len(avg_accuracies)):
        accuracy.append(avg_accuracies[i][0])

    plt.figure(3)
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(["0.0001","0.001","0.01","0.1","1","inf"],avg_times,"-o")
    plt.xlabel("C value")
    plt.ylabel("Average elapsed times in seconds")
    plt.title("Average elapsed time vs C value")
    plt.subplot(1,2,2)
    plt.bar(["0.0001","0.001","0.01","0.1","1","inf"],accuracy)
    plt.xlabel("C value")
    plt.ylabel("Average accuracy")
    plt.title("Average accuracy vs C")
    plt.show()

train = np.array(pd.read_csv("human-activity-recognition-with-smartphones/train.csv",low_memory=False))
train_labels = train.T[len(train.T)-1].T
train_features = train.T[0: len(train.T)-3].T

test = np.array(pd.read_csv("human-activity-recognition-with-smartphones/test.csv",low_memory=False))
test_labels = test.T[len(test.T)-1].T
test_features = test.T[0: len(test.T)-3].T

print(train_features.shape)
print(test_features.shape)

totalsamples = train_features.shape[0] + test_features.shape[0]
print("Number of samples: ",totalsamples)

all_features = np.concatenate((train_features,test_features))
all_labels = np.concatenate((train_labels,test_labels))

pca = PCA(n_components = 0.99)
principleComp = pca.fit_transform(all_features)

principleCompTrain = principleComp[0:7352]
principleCompTest = principleComp[7352:]

print(principleCompTrain.shape)
print(principleCompTest.shape)

print(pd.DataFrame(data=principleComp))

#Taking two samples for LAYING label
#imbalance_training_features = np.concatenate((principleCompTrain[train_labels!="LAYING"], principleCompTrain[train_labels=="LAYING"][0:500] ))
#imbalance_training_labels = np.concatenate((train_labels[train_labels!="LAYING"], train_labels[train_labels=="LAYING"][0:500]))
#print(imbalance_training_labels[imbalance_training_labels=="LAYING"])

#Kfold applied SVM with normal labels
#kFold_SVM(principleCompTrain,train_labels,"balanced")

#Kfold applied SVM with manipulated labels
kFold_SVM(principleCompTrain,train_labels,"unbalanced")
