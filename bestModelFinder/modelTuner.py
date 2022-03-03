import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


class tuneModel:
    def __init__(self):
        self.clfXGB = XGBClassifier()
        self.clfSVC = SVC()

    def getBestParamsforSVC(self, X_train, y_train):
        clfSVC = SVC()
        params = { 'kernel':['rbf','sigmoid'],
                    'C':[0.1, 0.5, 1],
                    'random_state':[0,100,200,300]
                   }
        gsCV = GridSearchCV(estimator=clfSVC,param_grid=params,cv=5,verbose=3)
        gsCV.fit(X_train,y_train)
        gsCVResult = pd.DataFrame(gsCV.cv_results_)
        gsCVResult.to_csv("bestModelFinder/SVCgsCVResults.csv", index=False)
        kernal = gsCV.best_params_['kernel']
        C = gsCV.best_params_['C']
        random_state = gsCV.best_params_['random_state']

        # Model fitting for best param
        clfSVC = SVC(kernel=kernal,C=C,random_state=random_state)
        clfSVC.fit(X_train,y_train)
        print('SVC Model Trained')
        return clfSVC

    def getBestParamsForXGBC(self,X_train,y_train):
        clfXGB = XGBClassifier()
        params = {"n_estimators": [100, 130],
                  "criterion": ['gini', 'entropy'],
                  "max_depth": range(8, 10, 1)
                  }
        gsCV = GridSearchCV(estimator=clfXGB, param_grid=params,
                            cv=5, verbose=1)
        gsCV.fit(X_train,y_train)
        gsCVResult = pd.DataFrame(gsCV.cv_results_)
        gsCVResult.to_csv("bestModelFinder/XGBgsCVResults.csv", index=False)
        # Getting best params
        citerion = gsCV.best_params_["criterion"]
        nEstm = gsCV.best_params_['n_estimators']
        maxDepth = gsCV.best_params_['max_depth']

        # fitting with Best Model & params
        clfXGB = XGBClassifier(criterion = citerion, max_depth=maxDepth,
                                    n_estimators=nEstm, n_jobs=-1, verbose=1)
        clfXGB.fit(X_train,y_train)
        print('XGB Model Trained')

        return clfXGB

    def getBestModel(self,X_train,X_test,y_train,y_test):
        # Getting scores for each model
        bestscore = 0
        clfSVC = self.getBestParamsforSVC(X_train, y_train)
        clfXGB = self.getBestParamsForXGBC(X_train, y_train)
        bestModel = clfXGB
        for model in (clfSVC,clfXGB):
            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            if y_test.nunique()==1:
                score = accuracy_score(y_test,y_predict)
            else:
                score = roc_auc_score(y_test,y_predict)
            bestscore, bestModel = (score, model) if score >= bestscore else (bestscore,bestModel)
            bestModelName = 'SVC' if bestModel == self.clfSVC else 'XGB'


        return bestModel, bestModelName