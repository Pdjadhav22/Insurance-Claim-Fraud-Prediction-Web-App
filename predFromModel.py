import pandas as pd
import numpy as np

from Data_Preprocessor.Data_preprocessor import preProcessing
from fileOperations.fileMethods import fileMethods

class predFromModel:
    def __init__(self):
        self.filepath = 'goodDataToPred/goodPredData.csv'

    def predModel(self):
        "Data Loading"
        preprocessor = preProcessing()
        data = preprocessor.loadData(self.filepath)

        "Data Preprocessing"
        # nonRelCols = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
        #               'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
        #               'auto_make', 'auto_model', 'auto_year', 'age', 'total_claim_amount']
        "Selecting choosen features only from dataset based EDA results"
        features = ['insured_hobbies', 'incident_type', 'collision_type', 'incident_severity',
            'authorities_contacted', 'incident_state', 'property_damage','umbrella_limit',
           'policy_deductable','number_of_vehicles_involved','witnesses',
            'policy_annual_premium','property_claim']
        data = preprocessor.removeColumns(data,features)
        data = preprocessor.removeWhiteSpaces(data)
        data = preprocessor.cleanup(data)
        data = preprocessor.imputeMissingValues(data)
        # data = preprocessor.scaledata(data)
        data = preprocessor.encodeCatcols(data)
        # X, y = preprocessor.seperateLabels(data)

        ## clustering removed
        # "clustering"
        # fileops = fileMethods()
        # model = fileops.modelLoader("Kmeans",'Kmeans')
        # cluster = model.predict(data)
        # data["Clusters"] = cluster
        # clusters = data["Clusters"].unique()
        # predictions = []
        # for cluster in clusters:
        #     clusterData = data[data["Clusters"]==cluster]
        #     clusterData = clusterData.drop(["Clusters"], axis=1)
        #     modelName = fileops.findBestModel((str(int(cluster))))
        #     print(modelName, " selected for #", cluster)
        #     model = fileops.modelLoader(modelName,str(int(cluster)))
        #     print(type(clusterData),clusterData.columns)
        #     clusterDataPred = model.predict(clusterData)

        
        fileops = fileMethods()
        # model = fileops.modelLoader("Kmeans",'Kmeans')
        # cluster = model.predict(data)
        # data["Clusters"] = cluster
        # clusters = data["Clusters"].unique()
        predictions = []
        # for cluster in clusters:
        # clusterData = data[data["Clusters"]==cluster]
        # clusterData = clusterData.drop(["Clusters"], axis=1)
        modelName = fileops.findBestModel()
        print(modelName, " selected ")
        model = fileops.modelLoader(modelName)
        # print(type(clusterData),clusterData.columns)
        # data_pred = model.predict(data)

        # applying manual threshold
        threshold=0.35
        data_pred = model.predict_proba(data)
        y_pred_th = np.where(data_pred[:,1]>threshold,1,0)
        
        for rec in y_pred_th:
            if rec == 0:
                predictions.append('N')
            else:
                predictions.append(("Y"))

        final = pd.DataFrame(predictions, columns=["predictions"])
        final.to_csv("predOutFile/Predictions.csv")
        return "Prediction Completed"


