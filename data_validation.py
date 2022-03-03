from  values_from_schema import schema_values
from Data_Transformation.Data_Transformation import  Data_transform
from DbOperations.dbOperations import dbOperations

class dataValidation:
    def __init__(self,path):
        self.schema_values = schema_values(path)

    def train_data_val(self):
        LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName = self.schema_values.valuesFromSchema()
        regex = self.schema_values.manualFileNameRegex()

        self.schema_values.fileGoodBadSegregate(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)

        # Data transformation
        data_tranform = Data_transform()
        data_tranform.dataTransform()

        #Db operations
        dbOps = dbOperations()
        dbOps.insertData(ColName)
        dbOps.extractDatafromDb()