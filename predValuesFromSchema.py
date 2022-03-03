import json
import os
import re
import shutil
from os import listdir

from values_from_schema import schema_values

class predValFromSchema:
    def __init__(self,path):
        self.batchDir = path
        self.schema_file = "input/schema_prediction.json"
        self.regex = r"fraudDetection_\d*_\d*.csv"

    def predValuesSchema(self):
        with open(self.schema_file,'r') as f:
            predDataCriteria = json.load(f)
            LengthOfDateStampInFile = predDataCriteria['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = predDataCriteria['LengthOfTimeStampInFile']
            NumberofColumns = predDataCriteria['NumberofColumns']
            ColName = predDataCriteria['ColName']

            return LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,ColName

    def predDataSegregate(self,LengthOfDateStampInFile, LengthOfTimeStampInFile):
        for file in listdir(self.batchDir):

            #File name check
            if re.match(self.regex, file):
                fileDateStamp = file.split('_')[1]
                fileTimeStamp = file.split('_')[2].split('.')[0]
                if (len(fileDateStamp) == LengthOfDateStampInFile) and (len(fileTimeStamp) == LengthOfTimeStampInFile):
                    print(file, ' is Good File')
                    self.predMoveGoodData(file)
                else:
                    print(file, ' is Bad file')
                    self.predMoveBadData(file)
            else:
                print(file, ' is Bad file')
                self.predMoveBadData(file)


    def predMoveGoodData(self,file):
        dataValPath = 'Data_val/predGoodData/'
        if not os.path.exists(dataValPath):
            os.makedirs(dataValPath)
            shutil.copy(self.batchDir+file,dataValPath )
        else:
            shutil.copy(self.batchDir + file, dataValPath)
        print('Good Files copied to Good location')

    def predMoveBadData(self,file):
        dataValPath = 'badPredDataArchive/'
        if not os.path.exists(dataValPath):
            os.makedirs(dataValPath)
            shutil.copy(self.batchDir+file, dataValPath)
        else:
            shutil.copy(self.batchDir + file, dataValPath)
        print("Bad file Moved to Bad location")



