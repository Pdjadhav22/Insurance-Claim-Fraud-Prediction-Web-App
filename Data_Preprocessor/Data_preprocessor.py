import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns


class preProcessing:
    def __init__(self):
        pass
        # self.inputFile= 'GoodDataForTrain/inputFile.csv'


    def loadData(self,path):
        self.data = pd.read_csv(path)
        print('data Loaded')
        return  self.data

    def removeColumns(self,data,cols):
        dataWremovedcols = data.drop(cols, axis=1)
        print('Specified columns are removed from Dataset')
        return  dataWremovedcols

    def removeWhiteSpaces(self,data):
        dataWremovedwhiteSpaces = data.apply(lambda x:x.str.strip() if x.dtype=='O' else x)
        print('Whitespaces removed')
        return  dataWremovedwhiteSpaces

    # handling "?" values in data
    def cleanup(self,data):
        dataCleanup = data.replace(to_replace='?', value= np.NAN)
        print('Data "?" replaced by NaN')
        return dataCleanup


    def imputeMissingValues(self,data):
        # check for cols with missing values
        colsWithMissingValues = data.isna().sum()[data.isna().sum()>0].index

        # Imputation of missing values
        numImputer = SimpleImputer()
        catImputer = SimpleImputer(strategy='most_frequent')
        for col in colsWithMissingValues:
            if data[col].dtype=='O':
                data[col] = catImputer.fit_transform(data[[col]])
            else:
                data[col] = numImputer.fit_transform(data[[[col]]])
        print('Missing values handleded')
        data.to_csv('Data_Preprocessor/MissingImpute.csv')
        return data

    def seperateLabels(self, data):
        X = data.drop('fraud_reported', axis=1)
        y = data['fraud_reported']
        print('Labels seperated')
        return X,y

    def scaledata(self,data):
        #applicable only for numeric data
        # Getting numeric Columns
        print('Data Scaled')
        data.to_csv('Data_Preprocessor/ScaledData.csv')
        return data
        # except Exception as e:
        #     print('Data contains Categorical Features\n',e)

    def encodeCatcols(self, data):
        catData = data.select_dtypes(include='O')

        # mapping know features manualy
        # Hard code Cat encoding based on EDA
        catData['policy_csl'] = catData['policy_csl'].map({'100/300': 1, '250/500': 2.5, '500/1000': 5})
        catData['insured_education_level'] = data['insured_education_level'].map(
            {'JD': 1, 'High School': 2, 'College': 3, 'Masters': 4, 'Associate': 5, 'MD': 6, 'PhD': 7})
        catData['incident_severity'] = catData['incident_severity'].map(
            {'Trivial Damage': 1, 'Minor Damage': 2, 'Major Damage': 3, 'Total Loss': 4})
        catData['insured_sex'] = catData['insured_sex'].map({'FEMALE': 0, 'MALE': 1})
        catData['property_damage'] = catData['property_damage'].map({'NO': 0, 'YES': 1})
        catData['police_report_available'] =catData['police_report_available'].map({'NO': 0, 'YES': 1})
        catDataMapped = ['policy_csl', 'insured_education_level','incident_severity', 'insured_sex', 'property_damage', 'police_report_available']

        # for Training dataset
        try:
            catData['fraud_reported'] = catData['fraud_reported'].map({'N': 0, 'Y': 1})
            catDataMapped = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                  'property_damage', 'police_report_available', 'fraud_reported']
        except:
            #for Prediction dataset
            catDataMapped = ['policy_csl', 'insured_education_level', 'incident_severity', 'insured_sex',
                                  'property_damage', 'police_report_available']



        #Encoding using Get Dummies
        # data = data.select_dtypes(exclude='O')
        catDataNotMapped = catData.select_dtypes(include='O').columns
        catData2 = pd.get_dummies(catData, columns=catDataNotMapped, drop_first=True)
        catData2 = catData2.select_dtypes(exclude='O')
        data = data.select_dtypes(exclude='O')
        data = pd.concat([data,catData2],axis=1)
        print('Catogorical featured Encoded')
        return data

    def handleImbalanceDataset(self, x, y):
        rosampler = RandomOverSampler()
        XSampled, ySampled = rosampler._fit_resample(x,y)
        print('Dataset Imbalance Handled')
        return XSampled, ySampled

    def plotdist(self,data):
        pairPlot = sns.pairplot(data, kind='hist')
        pairPlot.figure.savefig("PairPlot.png")
