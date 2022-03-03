import os.path
import pickle
import shutil
from os import listdir


class fileMethods:

    def __init__(self):
        self.modelDir = 'models/'

    def saveModel(self, model, fileName):
        loc = os.path.join(self.modelDir+fileName+'/')
        if os.path.isdir(loc):
            shutil.rmtree(loc)
            os.makedirs(loc)
        else:
            os.makedirs(loc)
        with open(loc+fileName+'.sav', 'wb') as f:
             pickle.dump(model ,f)
        return 'success'

    def modelLoader(self,filename):
        with open(self.modelDir+filename+'/'+filename+'.sav', 'rb') as f:
            return pickle.load(f)
        print(filename , " Model Loaded successfully")

    def findBestModel(self,clusteNumber):
        modelName = ''
        for dir in listdir(self.modelDir):
            modelNamesplit=str(dir).split('_')
            if len(modelNamesplit)>1 and int(modelNamesplit[1])==clusteNumber:
                modelName= dir
                break
        return modelName

