import os.path
import pickle
import shutil
from os import listdir


class fileMethods:

    def __init__(self):
        self.modelDir = 'models/'

    def saveModel(self, model, fileName, clusteNumber):
        loc = os.path.join(self.modelDir+clusteNumber+'/')       
        if os.path.isdir(loc):
            shutil.rmtree(loc)
            os.makedirs(loc)
        else:
            os.makedirs(loc)
        with open(loc+fileName+'.sav', 'wb') as f:
             pickle.dump(model ,f)
        return 'success'

    def modelLoader(self,filename, clusternumber):
        with open(self.modelDir+clusternumber+'/'+filename+'.sav', 'rb') as f:
            return pickle.load(f)
        print(filename , " Model Loaded successfully")

    def findBestModel(self,clusteNumber):
        modelName = ''
        for dir in listdir(self.modelDir+clusteNumber+'/'):
            # modelNamesplit=str(dir).split('_')
            # if len(modelNamesplit)>1 and int(modelNamesplit[1])==clusteNumber:
            dir = dir.split('.')[0]
            modelName= dir
            break
        return modelName

