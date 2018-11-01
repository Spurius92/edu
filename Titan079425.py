#Most of this code i copy-pasted from the other peoples kernels
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
sns.set(style="whitegrid", palette = 'muted')
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

path1 = '../input/train.csv'
train_data = pd.read_csv(path1, sep=',')  
path2 = '../input/test.csv'  # load the competition test data
test_data = pd.read_csv(path2, sep=',')
data = train_data.append(test_data, sort=False)   #Append rows of data2 to data1
train_len = len(train_data)

data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})  #set Sex in categorical values
data['Embarked'] = data['Embarked'].fillna('S')      #set missing Embarked with the most common S
data['FamSize'] = data.SibSp + data.Parch + 1       #New column 

data['Title'] = data.Name  # for now copy name directly
def extract_title(x):   # x is entire row
    string=x['Title']
    ix = string.find(".")    # use .find to find the first dot
    for i in range(0,ix):
        if (string[ix-i] == ' '):  # if we find space, then stop iterating
            break                   # break out of for-loop
    return string[(ix-i+1):ix]  # return everything after space up till before the dot
data['Title']=data.apply(extract_title, axis=1)
 
index_NaN_age = list(data["Age"][data["Age"].isnull()].index)         
for i in index_NaN_age :                                   #substitute nans in 'Age' with median values
    age_med = data["Age"].median()
    age_pred = data["Age"][((data['SibSp'] == data.iloc[i]["SibSp"]) & 
                               (data['Parch'] == data.iloc[i]["Parch"]) & 
                               (data['Pclass'] == data.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        data['Age'].loc[i] = age_pred
    else :
        data['Age'].loc[i] = age_med

data['Fare'] = data['Fare'].fillna(data['Fare'].median())       #working with ['Fare']
data['Fare'] = data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
data["Title"] = data["Title"].astype(int)
data.drop(labels=['Name'], axis=1, inplace=True)  # we can even drop the 'Name' column now 

data["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in data['Cabin'] ])

Ticket = []
for i in list(data.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
data["Ticket"] = Ticket
data["Ticket"].head()
data = pd.get_dummies(data, columns = ["Ticket"], prefix="T")
data["Pclass"] = data["Pclass"].astype("category")
data = pd.get_dummies(data, columns = ["Pclass"],prefix="Pc")
data = pd.get_dummies(data, columns = ["Title"],prefix="Title")
data = pd.get_dummies(data, columns = ["Embarked"],prefix="Emb")
data = pd.get_dummies(data, columns = ["Cabin"],prefix="Cabin")

train = data[:train_len]
test = data[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)
IDtest = test["PassengerId"]  
#print(test.isnull().sum())        temporary here


kfold = StratifiedKFold(n_splits=10)      #some comparative tool for different algorithms
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold,))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

RFC = RandomForestClassifier(n_estimators=200, max_depth=2,
                              random_state=0, n_jobs = 4, verbose = 1 )
RFCmodel = RFC.fit(X_train, Y_train)
Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 5, 15],
              "min_samples_split": [1, 5, 15],
              "min_samples_leaf": [1, 5, 15],
              "bootstrap": [False],
              "n_estimators" :[100, 300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_

DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_

ExtC = ExtraTreesClassifier()
# Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train,Y_train)
ExtC_best = gsExtC.best_estimator_
# Best score
gsExtC.best_score_

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,Y_train)
GBC_best = gsGBC.best_estimator_
# Best score
gsGBC.best_score_

### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train,Y_train)
SVMC_best = gsSVMC.best_estimator_
# Best score
gsSVMC.best_score_


test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)

votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)     #Final modeling and submitting
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("Titanic-mitanic.csv",index=False)
