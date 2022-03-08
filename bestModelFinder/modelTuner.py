import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


class tuneModel:
    def __init__(self):
        self.clfXGB = XGBClassifier()
        self.clfRF = RandomForestClassifier()

    def getBestParamsforRF(self, X_train, y_train):
        clfRF = RandomForestClassifier()
        params = { 'n_estimators':[100,150,200],
                    'criterion':['gini','entropy'],
                    'random_state':[0,100,200,300]
                   }
        gsCV = GridSearchCV(estimator=clfRF,param_grid=params,scoring='recall',cv=5,verbose=3)
        gsCV.fit(X_train,y_train)
        gsCVResult = pd.DataFrame(gsCV.cv_results_)
        gsCVResult.to_csv("bestModelFinder/RFGCVresults.csv", index=False)
        n_estimators = gsCV.best_params_['n_estimators']
        criterion = gsCV.best_params_['criterion']
        random_state = gsCV.best_params_['random_state']

        # Model fitting for best param
        clfRF = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,random_state=random_state)
        clfRF.fit(X_train,y_train)
        print('RandomForest Model Trained')
        return clfRF

    def getBestParamsForXGBC(self,X_train,y_train):
        clfXGB = XGBClassifier()
        params = {"n_estimators": [100, 130],
                  "criterion": ['gini', 'entropy'],
                  "max_depth": range(8, 10, 1)
                  }
        gsCV = GridSearchCV(estimator=clfXGB, param_grid=params,scoring='recall',
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
        clfRF = self.getBestParamsforRF(X_train, y_train)
        clfXGB = self.getBestParamsForXGBC(X_train, y_train)
        bestModel = clfXGB
        for model in (clfRF,clfXGB):
            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            if y_test.nunique()==1:
                score = accuracy_score(y_test,y_predict)
            else:
                score = recall_score(y_test,y_predict)
            plt.figure(figsize=(5,5))
            sns.heatmap(confusion_matrix(y_test,y_predict),annot=True, cbar=False, fmt='g', cmap="crest")
            plt.savefig('bestModelFinder/confusionMatrix.png')
            bestscore, bestModel = (score, model) if score >= bestscore else (bestscore,bestModel)
            bestModelName = 'RF' if bestModel == self.clfRF else 'XGB'


        return bestModel, bestModelName