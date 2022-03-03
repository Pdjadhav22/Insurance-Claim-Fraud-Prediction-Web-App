import pandas as pd
from os import listdir

class Data_transform:
    def __init__(self):
        self.goodFilePath = "Data_val/Good_raw/"

    def dataTransform(self):
        for file in listdir(self.goodFilePath):
            data = pd.read_csv(self.goodFilePath+'/'+file)
            cat = data.select_dtypes(include='O')
            cat_cols = cat.columns
            for col in cat_cols:
                data[col] = data[col].apply(lambda x:"'"+str(x)+"'")
            # print(data)
