import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

failure = pd.read_csv('heart_failure_clinical_records_dataset.csv')
#print(failure.head(5))

x = failure.values
#print(x)

check = failure.isna().sum()
check_boolean = failure.isna().any().any()
#print(check_boolean)

data_0 = failure[failure['DEATH_EVENT']==0]
data_1 = failure[failure['DEATH_EVENT']==1]

def variableCorrelation():
    corr = failure.corr()
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, annot=True, square=True)
    plt.show()
    #Sex and Smoking has a big correlation
#print(variableCorrelation())

def ages():
    death_yes = failure[failure['DEATH_EVENT']==1]
    death_no = failure[failure['DEATH_EVENT']==0]
    ages_death = death_yes['age']
    ages_survived = death_no['age']
    #print(ages_death)
    sns.displot(failure, x=ages_death)
    sns.displot(failure, x=ages_survived)
    plt.show()
#print(ages())

#Feature Selection for training and testing
def FeatureImportance():
    x = failure[failure.columns[:-1]]
    y = failure.iloc[:,-1]
    tree_feature = ExtraTreesClassifier()
    tree_feature.fit(x,y)
    feat_importance = pd.Series(tree_feature.feature_importances_,index=x.columns)
    print(feat_importance)
    feat_importance.nlargest(10).plot(kind='barh')
    plt.show()
#print(FeatureImportance())


#Target: DEATH_EVENT
#Features: smoking, sex, age, anaemia, etc.

features = failure.iloc[:,[4,7,11]]
#print(features)
target = failure.iloc[:,-1]
#something = failure.iloc[:,:-1]
#print(something)
#print(features)

x_train,x_test,y_train,y_test = train_test_split(features,target, train_size=0.3, random_state=30)

standard = StandardScaler()
x_train_scaled = standard.fit_transform(x_train)
x_test_scaled = standard.fit_transform(x_test)

def bestParametersLogistic():
    log = LogisticRegression()
    parameters = [{
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
    }]
    log_search = GridSearchCV(log,parameters,cv=5,scoring='accuracy')
    log_search.fit(x_train_scaled,y_train)
    print(log_search.best_params_)
#print(bestParametersLogistic())


def logistic():
    logist = LogisticRegression(penalty='l1', solver='liblinear', C=1)
    logist.fit(x_train_scaled,y_train)
    log_predict = logist.predict(x_test_scaled)
    log_score = accuracy_score(y_test,log_predict)*100   #85%
    print(log_predict)
    print(log_score)
    
    #Use the ROC/AUC curve to find the Score 
    log_prob = logist.predict_proba(x_test_scaled)[:,1]
    false_positive_log,true_positive_log,threshold_log = roc_curve(y_test,log_prob)
    auc_score = metrics.auc(false_positive_log,true_positive_log)
    print("AUC:", auc_score)
    plt.plot(false_positive_forest,true_positive_forest)
    plt.plot([0,1], ls='--')
print(logistic())

#Find the Best K-neighbor score to use
def KNNscores():
    scores = []
    for k in range(3,20):
        neighbor = KNeighborsClassifier(n_neighbors=k, metric='minkowski')
        neighbor.fit(x_train_scaled,y_train)
        k_predicted = neighbor.predict(x_test_scaled)
        scores.append(accuracy_score(y_test,k_predicted))
    plt.plot(range(3,20),scores)
    plt.show()
#print(KNNscores())

def KNN():
    k = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    k.fit(x_train_scaled,y_train)
    knn_pred = k.predict(x_test_scaled)
    knn_score = accuracy_score(y_test,knn_pred)
    print(knn_pred)
    print(knn_score)
print(KNN())

def KNNGridSearch():
    Knn = KNeighborsClassifier()
    param_grid = {
        'C':[0.01,0.1,1,10,100],
        'n_neighbors': list(range(2,18)),
        'p':[1,2,3,4,5,6,7,8,9,10],
        'weights': ['uniform','distance']
    }
    model = GridSearchCV(Knn,param_grid=param_grid,scoring='accuracy')
print(KNNGridSearch())


