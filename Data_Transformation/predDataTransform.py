import numpy as np
import pandas as pd
from os import listdir


class predDataTransform:
    def __init__(self):
        self.predGoodData = "Data_val/predGoodData/"

    def removeWhitespaces(self):
        for file in listdir(self.predGoodData):
            data = pd.read_csv(self.predGoodData+file)
            catcols = data.select_dtypes(include="O").columns
            for col in catcols:
                data[col]=data[col].apply(lambda x: str(x).strip())

            # replace ? with Null values
            data.replace("?",np.NAN, inplace=True)

            data.to_csv(self.predGoodData+file, index=None, header=True)
            print("Whitespaces removed from data from all good files")


