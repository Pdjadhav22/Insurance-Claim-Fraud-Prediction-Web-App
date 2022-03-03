import csv
import os.path
import shutil
import sqlite3
from os import listdir

import pandas as pd


class predDbOps:
    def __init__(self):
        self.dbPath = "predDbOperation/"
        self.predGoodData = "Data_val/predGoodData/"

    def conn(self):
        conn = sqlite3.connect(self.dbPath+'PredctionDB.db')
        return conn

    def createTable(self,cols):
        conn = self.conn()
        cur = conn.cursor()
        c=0
        for c,(key,value) in enumerate(cols.items()): # looping through all cols to create col one by one
            key = '_'.join(key.split('-')) if '-' in key else key
            try:
                q1 = f"alter table predGoodData add {key} {value}"
                cur.execute(q1)
                conn.commit()
            except:
                q1 = f"create table if not exists predGoodData ({key} {value})"
                cur.execute(q1)
        print(f"Table created with {c+1} Columns")

    def insertData(self):
        conn = self.conn()
        cur = conn.cursor()

        #Temp truncate Table for Dve
        q1 = "delete from predGoodData"
        cur.execute(q1)
        conn.commit()
        print('Old records Deleted')

        for file in listdir(self.predGoodData):
            with open(self.predGoodData+file, 'r') as f:
                next(f)
                data = csv.reader(f, delimiter = ',')
                for rec in data:
                    rec = ','.join(f"'{i}'" for i in rec)
                    q1= f"insert into predGoodData values ({rec})"
                    cur.execute(q1)
                    conn.commit()
        print("Prediction Data Inserted")
        shutil.rmtree(self.predGoodData)

    def extractPredData(self):
        conn = self.conn()
        cur = conn.cursor()
        goodPredData= "goodDataToPred/"
        data = pd.read_sql_query("select * from predGoodData", conn)
        if not os.path.exists(goodPredData):
            os.makedirs(goodPredData)
        data.to_csv(goodPredData+"goodPredData.csv", index=False)
        print("Data Extracted from Database")