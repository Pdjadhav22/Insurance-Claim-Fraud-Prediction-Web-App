from Data_Preprocessor.Data_preprocessor import preProcessing
from Data_Preprocessor.clustering import kMeansClustring
from sklearn.model_selection import train_test_split
from bestModelFinder.modelTuner import tuneModel
from fileOperations.fileMethods import fileMethods

class trainingModel:
    def __init__(self):
        self.filePath = 'GoodDataForTrain/inputFile.csv'

    def trainingModel(self):

        "Getting dataset from source"
        preprocessor = preProcessing()
        data = preprocessor.loadData(self.filePath)

        "Data Preprocessing"
        # remove non-relevent cols
        nonRelCols = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location',
                      'incident_date','incident_state','incident_city','insured_hobbies',
                      'auto_make','auto_model','auto_year','age','total_claim_amount']
        data = preprocessor.removeColumns(data,nonRelCols)
        data = preprocessor.removeWhiteSpaces(data)
        data = preprocessor.cleanup(data)
        data = preprocessor.imputeMissingValues(data)
        data = preprocessor.scaledata(data)
        data = preprocessor.encodeCatcols(data)
        X,y = preprocessor.seperateLabels(data)

        # preprocessor.plotdist(X)

        # Temp - Data imbalance need to handle
        # X,y = handleImbalanceDataset(X,y)

        "Applying Clustering Approach"
        cluster = kMeansClustring()
        cluster.elbowPlot(X)
        knee = cluster.getKnee(X)
        X = cluster.createClusters(X,knee)

        # "Parsing over all clusters to find best model for each cluster"
        # # adding label col to X for ease of split base on cluster
        X['labels'] = y
        for i in X['Cluster'].unique():
            print(i)
            df = X[X.Cluster==i]
            x = df.drop(['labels','Cluster'],axis=1)
            y = df.labels

            X_train,X_val,y_train,y_val = train_test_split(x,y,train_size=0.8,random_state=40)

            # Getting best Model
            modelFinder = tuneModel()
            bestModel, bestModelName = modelFinder.getBestModel(X_train,X_val,y_train,y_val)

            # Model Save
            fileOps = fileMethods()
            fileOps.saveModel(bestModel,bestModelName+'_'+str(i))
            print('Model Saved based on clusters in Model Folder')
        #





