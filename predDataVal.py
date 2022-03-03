from predValuesFromSchema import predValFromSchema
from Data_Transformation.predDataTransform import predDataTransform
from predDbOperation.predDbOperation import predDbOps

class predDataVal:
    def __init__(self, path):
        self.path = path
        self.predGoodData = "Data_val/preGoodData/"

    def predDatavalInsertion(self):
        valuesFromSchema = predValFromSchema(self.path)
        LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,ColName = valuesFromSchema.predValuesSchema()
        valuesFromSchema.predDataSegregate(LengthOfDateStampInFile, LengthOfTimeStampInFile)

        # For each file in good data
        # Data Trasnform to ease of upload in DB
        presTransfom = predDataTransform()
        presTransfom.removeWhitespaces()

        # DB operation to upload data in Prediction Database
        predDB = predDbOps()
        predDB.createTable(ColName)
        predDB.insertData()
        predDB.extractPredData()
