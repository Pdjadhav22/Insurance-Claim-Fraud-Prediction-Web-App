import pandas as pd

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
        nonRelCols = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
                      'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
                      'auto_make', 'auto_model', 'auto_year', 'age', 'total_claim_amount']
        data = preprocessor.removeColumns(data,nonRelCols)
        data = preprocessor.removeWhiteSpaces(data)
        data = preprocessor.cleanup(data)
        data = preprocessor.imputeMissingValues(data)
        data = preprocessor.scaledata(data)
        data = preprocessor.encodeCatcols(data)
        # X, y = preprocessor.seperateLabels(data)

        "clustering"
        fileops = fileMethods()
        model = fileops.modelLoader("Kmeans")
        cluster = model.predict(data)
        data["Clusters"] = cluster
        clusters = data["Clusters"].unique()
        predictions = []
        for cluster in clusters:
            clusterData = data[data["Clusters"]==cluster]
            clusterData = clusterData.drop(["Clusters"], axis=1)
            modelName = fileops.findBestModel(cluster)
            print(modelName, " selected for #", cluster)
            model = fileops.modelLoader(modelName)
            clusterDataPred = model.predict(clusterData)
            for rec in clusterDataPred:
                if rec == 0:
                    predictions.append('N')
                else:
                    predictions.append(("Y"))

        final = pd.DataFrame(predictions, columns=["predictions"])
        final.to_csv("predOutFile/Predictions.csv")
        print("Prediction Completed")


