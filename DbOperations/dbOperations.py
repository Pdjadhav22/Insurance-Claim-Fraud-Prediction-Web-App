import collections.abc
import os.path
import shutil
import sqlite3
from os import listdir
import csv

import pandas as pd


class dbOperations:
    def __init__(self):
        self.path = 'DbOperations/'
        self.goodFiles = 'Data_val/Good_raw/'
        self.database = 'GoodFilteredData'

    def sqlConnect(self):
        conn =sqlite3.connect(self.path+self.database+'.db')
        return conn

    def createTable(self,cols):
        conn = self.sqlConnect()
        cur = conn.cursor()
        c=0

        # Temproroy Drop table for development, comment it out before finalising
        q1 = 'drop table if exists GoodRawData'
        cur.execute(q1)

        ####
        for c ,(key, dtype) in enumerate(cols.items()):
            if '-' in key:
                key = '_'.join(key.split('-'))
            try:
                q1 = f"alter table GoodRawData add {key} {dtype}"
                cur.execute(q1)
                conn.commit()

            except:
                q1 = f"create table if not exists GoodRawData ({key} {dtype})"
                cur.execute(q1)
                conn.commit()
        print(f'Table Created for {c+1} columns')

    def insertData(self, cols):
        self.createTable(cols)
        conn = self.sqlConnect()
        cur = conn.cursor()
        goodfilepath = self.goodFiles
        for file in listdir(goodfilepath):
            with open(goodfilepath+'/'+file,'r') as f:
                next(f)
                data = csv.reader(f, delimiter='\n')
                for rec in data:
                    rec = ','.join(["'"+i+"'" for i in rec[0].split(',')])
                    # rec = "','".join(rec.split(','))
                    q1 = f"insert into GoodRawData values ({rec})"
                    cur.execute(q1)
                    conn.commit()
        print('data inserted')
        shutil.rmtree(goodfilepath)

    def extractDatafromDb(self):
        self.extractPath = 'GoodDataForTrain/'
        self.extractFileName = 'inputFile.csv'

        conn = self.sqlConnect()
        data = pd.read_sql_query('select * from GoodRawData',conn)

        if not os.path.isdir('GoodDataForTrain/'):
            os.makedirs(self.extractPath)
        data.to_csv(self.extractPath+self.extractFileName,index=False)
        print ('Data Extracted for Model Training')
