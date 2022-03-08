from flask import request
import pickle
import pandas as pd
from Data_Preprocessor.Data_preprocessor import preProcessing
from fileOperations.fileMethods import fileMethods

class predFromRec:
    def __init__(self):
        pass

    def getValues(self):
    #    months_as_customer =  request.form['months_as_customer']
    #    policy_csl = request.form['policy_csl']
       policy_deductable = request.form['policy_deductable']
       policy_annual_premium = request.form['policy_annual_premium']
       umbrella_limit = request.form['umbrella_limit']
    #    insured_sex = request.form['insured_sex']
    #    insured_education_level = request.form['insured_education_level']
    #    insured_occupation = request.form['insured_occupation']
    #    insured_relationship = request.form['insured_relationship']
    #    capital_gains = request.form['capital_gains']
    #    capital_loss = request.form['capital_loss']
       incident_type = request.form['incident_type']
    #    collision_type = request.form['collision_type']
       incident_severity = request.form['incident_severity']
       authorities_contacted = request.form['authorities_contacted']
    #    incident_hour_of_the_day = request.form['incident_hour_of_the_day']
       number_of_vehicles_involved = request.form['number_of_vehicles_involved']
    #    property_damage = request.form['property_damage']
       bodily_injuries = request.form['bodily_injuries']
       witnesses = request.form['witnesses']
    #    police_report_available = request.form['police_report_available']
    #    injury_claim = request.form['injury_claim']
       property_claim = request.form['property_claim']
       insured_hobbies = request.form['insured_hobbies']
       incident_state = request.form['incident_state']
    #    vehicle_claim = request.form['vehicle_claim']

       featureDict = {
        #    'months_as_customer':int(months_as_customer),
        #    'policy_csl' : policy_csl,
           'policy_deductable':int(policy_deductable),
           'policy_annual_premium':float(policy_annual_premium),
           'umbrella_limit' : int(umbrella_limit),
        #    'insured_sex': insured_sex,
        #    'insured_education_level': insured_education_level,
        #    'insured_occupation': insured_occupation,
        #    'insured_relationship': insured_relationship,
        #    'capital_gains' : int(capital_gains),
        #    'capital_loss': -1*int(capital_loss),
           'incident_type': incident_type,
        #    'collision_type' : collision_type,
           'incident_severity' : incident_severity,
           'authorities_contacted': authorities_contacted,
        #    'incident_hour_of_the_day': int(incident_hour_of_the_day),
           'number_of_vehicles_involved': int(number_of_vehicles_involved),
        #    'property_damage': property_damage,
           'bodily_injuries': int(bodily_injuries),
           'witnesses': int(witnesses),
        #    'police_report_available': police_report_available,
        #    'injury_claim': int(injury_claim), 
           'property_claim': int(property_claim),
            # 'vehicle_claim': int(vehicle_claim),
            'insured_hobbies': insured_hobbies,
            'incident_state': incident_state,
           }
    
       return featureDict

    def predFromRec(self,rec):
        filepath = 'goodDataToPred/goodPredData.csv'
        preprocessor = preProcessing()
        data = preprocessor.loadData(filepath)
       
        "Data Preprocessing"
      #   nonRelCols = ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location',
      #               'incident_date', 'incident_state', 'incident_city', 'insured_hobbies',
      #               'auto_make', 'auto_model', 'auto_year', 'age', 'total_claim_amount']
        "Selecting choosen features only from dataset based EDA results"
        features = ['incident_type','incident_severity','authorities_contacted',
            'incident_state','policy_annual_premium','property_claim','policy_deductable', 
            'umbrella_limit','number_of_vehicles_involved', 'bodily_injuries', 'witnesses']
        data = preprocessor.removeColumns(data,features)
        print(data.shape[0])
        data = data.append(rec, ignore_index=True, sort=False)
        print(data.shape[0])
        print(data.columns)
        data = preprocessor.removeWhiteSpaces(data)
        data = preprocessor.cleanup(data)
        data = preprocessor.imputeMissingValues(data)
        data = preprocessor.scaledata(data)
        data = preprocessor.encodeCatcols(data)
        print(data.shape)
        data.to_csv("Data_Preprocessor/dataSinglerec.csv")

        "clustering"
        fileops = fileMethods()
        model = fileops.modelLoader("Kmeans",'Kmeans')
        cluster = model.predict(data)
        data["Clusters"] = cluster
        
        "Selecting user input record from processed data"
        predRec = pd.DataFrame()
        data = data.iloc[-1,:]
        cluster = data.Clusters
        data = data.drop('Clusters')
        modelName = fileops.findBestModel(str(int(cluster)))
        print(modelName, " selected for #", cluster)
        model = fileops.modelLoader(modelName,str(int(cluster)))
        # print('Before predict', data.columns)
        predRec = predRec.append(data, ignore_index=True, sort=False)
        print(type(predRec), predRec.columns)
        clusterDataPred = model.predict(predRec)
        for rec in clusterDataPred:
            if rec == 0:
                return 'No'
            else:
                return 'Yes'