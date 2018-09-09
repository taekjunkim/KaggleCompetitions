#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:03:51 2018

@author: taekjunkim
"""
#%% import modules
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;  ### good to visualize pandas dataframe

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


#%% load data files
TrainDF = pd.read_csv('./csvFiles/train.csv');
TrainDF.info();
TestDF = pd.read_csv('./csvFiles/test.csv');
TestDF.info();

### Columns
### PassengerId, Survived, Pclass, Name, Sex, Age, SibSp
### Parch, Ticket, Fare, Cabin, Embarked

### Let's get general idea how individual
### columns are related to Survived

#%% 1. Pclass - Categorical (three), but can be used to linear
print(TrainDF['Pclass'].value_counts().sort_index()); ### categorical 1,2,3
# linear relationship found between survival rates for Pclass
print(TrainDF['Survived'].groupby(TrainDF['Pclass']).mean()); 
sns.barplot(TrainDF['Pclass'],TrainDF['Survived']);

#%% 2. Sex - Categorical (two)
# Female has higher chances to be survived
print(TrainDF['Survived'].groupby(TrainDF['Sex']).mean());
# Regardless of Pclass
print(TrainDF['Survived'].groupby([TrainDF['Pclass'],TrainDF['Sex']]).mean());
sns.barplot(TrainDF['Pclass'],TrainDF['Survived'],hue=TrainDF['Sex']);

#%% 3. Age - Continuous for female, but not for male (see below)
AgeHisto = np.empty([4,8]);
for i in range(4):
    plt.subplot(2,2,i+1);
    SexIndex = np.ceil((i+1)/2)-1;
    if SexIndex==0:
        Sex = 'male';
    elif SexIndex==1:
        Sex = 'female';
    Surv = (i+1)%2;
    AgeNow = TrainDF['Age'][(TrainDF['Sex']==Sex) &
                            (TrainDF['Survived']==Surv)];
    ValNow = plt.hist(AgeNow.values, bins=8, range=[0,80]);
    AgeHisto[i,:] = ValNow[0];
    plt.title(['Survived: '+str(Surv)+' ,Sex: '+Sex]);

#%% Proportion of survivors, we found a linear relationship with Age
SurProp = np.empty([2,8]);
for i in range(2):
    row1 = (i+1)*2-2;
    row2 = (i+1)*2-1;
    SurProp[i,:] = AgeHisto[row1,:]/(AgeHisto[row1,:]+AgeHisto[row2,:]);
#SurProp[np.isnan(SurProp)]=0;    
plt.plot(np.arange(5,85,10),SurProp.T);
plt.xlabel('Age');
plt.ylabel('Survival rate');

#%% 4. SibSp & Parch 
print(TrainDF['Survived'].groupby(TrainDF['SibSp']).mean()); 
print(TrainDF['Survived'].groupby(TrainDF['Parch']).mean()); 
### Similar non-linear trend was found, so combine two varibles into FamSize
TrainDF['FamSize'] = TrainDF['SibSp']+TrainDF['Parch']+1;
TestDF['FamSize'] = TestDF['SibSp']+TestDF['Parch']+1;
print(TrainDF['Survived'].groupby(TrainDF['FamSize']).mean()); 
       
#%% Fare can help us to estimate Pclass
### Embarked 'C' has a higher survival rates than the others
### It seemed to be due to its relationship to Pclass
### Cabin information is insufficient
TrainDF = TrainDF.replace({'Embarked': {'C':0, 'Q':1, 'S':2}});
TrainDF['Embarked'][TrainDF['Embarked'].isnull()==1] = 0;
TrainDF['Embarked'] = TrainDF['Embarked'].astype(int);
TestDF = TestDF.replace({'Embarked': {'C':0, 'Q':1, 'S':2}});
TestDF['Embarked'] = TestDF['Embarked'].astype(int);
      

#%% Pclass, Sex, Age, Family size --> They are useful information
### We need to estimate Age for missing data
AllDF = pd.concat([TrainDF,TestDF],ignore_index=True);
### Age may be estimated from Title 
Temp = list(map(lambda x: x.split(',')[1].split('.')[0][1:], AllDF['Name'].values));
AllDF['Title'] = Temp;
AllDF['Title'][AllDF['Title'].values=='Capt'] = 'Sac';
AllDF['Title'][AllDF['Title'].values=='Col'] = 'Mr';
AllDF['Title'][AllDF['Title'].values=='Don'] = 'Mr';
AllDF['Title'][AllDF['Title'].values=='Jonkheer'] = 'Mr';
AllDF['Title'][AllDF['Title'].values=='Major'] = 'Mr';
AllDF['Title'][AllDF['Title'].values=='Rev'] = 'Sac';
AllDF['Title'][AllDF['Title'].values=='Sir'] = 'Mr';
AllDF['Title'][AllDF['Title'].values=='the Countess'] = 'Mrs';
AllDF['Title'][AllDF['Title'].values=='Dona'] = 'Mrs';
AllDF['Title'][AllDF['Title'].values=='Mme'] = 'Mrs';
AllDF['Title'][AllDF['Title'].values=='Lady'] = 'Mrs';
AllDF['Title'][AllDF['Title'].values=='Mlle'] = 'Mrs';
AllDF['Title'][AllDF['Title'].values=='Ms'] = 'Mrs';

AllDF['Sex'][AllDF['Sex'].values=='male'] = 0;
AllDF['Sex'][AllDF['Sex'].values=='female'] = 1;

MissingAges = np.where(AllDF['Age'].isnull()==1)[0];
for i in range(len(MissingAges)):
    idxNow = MissingAges[i];
    MatchingIdx = np.where(AllDF['Title'].values==AllDF['Title'][idxNow])[0];
    AllDF['Age'][idxNow] = AllDF['Age'][MatchingIdx].mean();                           
AllDF['Title'][AllDF['Age'].isnull()].value_counts();

#%% In female, survival rate increased with ages
### In male, survival rate is high only for <10 
kkk = np.where((AllDF['Sex'].values==0) 
               & (AllDF['Age'].values<10))[0];
AllDF['IsBoy'] = np.zeros(AllDF.shape[0],dtype=int);
AllDF['IsBoy'][kkk] = 1;

#%% HasCabin
HasCabin = np.where(AllDF['Cabin'].isnull()==0)[0];
NoCabin = np.where(AllDF['Cabin'].isnull()==1)[0];
AllDF['Cabin'][HasCabin] = 1;
AllDF['Cabin'][NoCabin] = 0;

#%% Make Train & Test Matrix
AllDF = pd.get_dummies(AllDF, columns=['Embarked']);
AllDF = pd.get_dummies(AllDF, columns=['Title']);

AllDF['XSurvived'] = AllDF['Survived'];
AllDF.drop(['Survived','FamSize','Fare','Name','Ticket','PassengerId'],
           axis=1,inplace=True);
ColumnName = list(AllDF.columns.values);
ColumnName.sort();
AllDF = AllDF[ColumnName];

TrainMtx = AllDF.iloc[0:890,:];
TestMtx = AllDF.iloc[891:,:-1];

TrainX = TrainMtx.iloc[:,:-1].values;
TrainY = TrainMtx['XSurvived'].values;

#%% Model selection                          
clf = RandomForestClassifier();
param_grid = {'n_estimators':[100,200,400],'min_samples_split':[2,4,8]};
cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid,cv=3);
cv_clf.fit(TrainX,TrainY);
clf.n_estimators = cv_clf.best_params_['n_estimators'];
clf.min_samples_split = cv_clf.best_params_['min_samples_split'];
aaa = cross_val_score(clf,TrainX,y=TrainY);

"""
clf = GradientBoostingClassifier();
param_grid = {'max_depth':[3,4,5],'n_estimators':[100,200,300,400]};
cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid,cv=3);
cv_clf.fit(TrainX,TrainY);
clf.max_depth = cv_clf.best_params_['max_depth'];
clf.n_estimators = cv_clf.best_params_['n_estimators'];
aaa = cross_val_score(clf,TrainX,y=TrainY);

clf = SVC();
param_grid = {'C':np.logspace(-2,2,10)};
cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid,cv=3);
cv_clf.fit(TrainX,TrainY);
clf.C = cv_clf.best_params_['C'];
aaa = cross_val_score(clf,TrainX,y=TrainY);
"""

TestX = TestMtx.iloc[:,:].values;
clf.fit(TrainX,TrainY);


Submitted = pd.DataFrame();
Submitted['PassengerId'] = TestDF['PassengerId'];
Submitted['Survived'] = clf.predict(TestX).astype(int);
Submitted.to_csv("TitanicSubmission.csv", index=False)
