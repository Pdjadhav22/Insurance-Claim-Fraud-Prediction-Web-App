from flask import Flask, render_template,request
import os
from data_validation import dataValidation
from trainModel import trainingModel
from predDataVal import predDataVal
from predFromModel import predFromModel
from predSingleEntry import predFromRec

import  warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def trainRouteClient():
    # get input train file path
    paths = "D:\Learning\Data Science\E2E Project\insuranceFraudDetection\Self\Training batch files"+ "/"

    # Data Validation : Checking input data against input schema (as per agreement with client)
    if os.path.isdir(paths):

        # Input Batch file validate & export from DB for Training
        dataValidate = dataValidation(paths)
        dataValidate.train_data_val()

        # Model training
        modelTrain = trainingModel()
        modelTrain.trainingModel()


    else: print('Bad path dir')

# trainRouteClient()

@app.route('/predict', methods=['POST'])
def predictRouteClient():
    if request.method == 'POST':
        path = request.form['filepath']
            # path  = 'D:\Learning\Data Science\E2E Project\insuranceFraudDetection\Self\PredictionBatchFile'+ "/"
        
        if os.path.isdir(path):
            
            #Prediction Data Validation
            predictDataValid = predDataVal(path)
            predictDataValid.predDatavalInsertion()

            #Predicitons
            predicitor = predFromModel()
            res = predicitor.predModel()

            return render_template('index.html',
             batchPredictionText='''Predcition Completed
            Output saved in predOutFile/Prediction.csv''')
            return res

@app.route('/singleRec', methods=['POST'])
def predRec():
    if request.method=='POST':         
        predSinglerec = predFromRec()
        dictData = predSinglerec.getValues()
        res = predSinglerec.predFromRec(dictData)

        if (res == 'No'):
            return render_template('index.html', prediction_text='No, This Claim is not Fraud')
    

        else:
            return render_template('index.html', prediction_text= 'Yes, This calim is a Fraud')
            # return res

# predRec()
# predictRouteClient()

if __name__ == '__main__':
    app.run(debug=True)