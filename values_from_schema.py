import json
import os.path
import re
import shutil
from os import  listdir


class schema_values():
    def __init__(self,path):
        self.batch_directory = path
        self.schema_file = "./input/schema_training.json"
        # Need to add Logger

    def valuesFromSchema(self):
        with open(self.schema_file,'r') as f:
            criteria_dic = json.load(f)
        LengthOfDateStampInFile = criteria_dic['LengthOfDateStampInFile']
        LengthOfTimeStampInFile = criteria_dic['LengthOfTimeStampInFile']
        NumberofColumns = criteria_dic['NumberofColumns']
        ColName = criteria_dic['ColName']

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, ColName

    def manualFileNameRegex(self):
        regex = 'fraudDetection_\d*_\d*.csv'
        return regex

    def fileGoodBadSegregate(self, regex,LengthOfDateStampInFile, LengthOfTimeStampInFile):
        # input data receives in Batch of files
        files = listdir(self.batch_directory)
        for file in files:
            if re.match(regex,file):
                fileNameSplit = file.split('.csv')[0].split('_')
                if len(fileNameSplit[1])== LengthOfDateStampInFile:
                    if len(fileNameSplit[2])== LengthOfTimeStampInFile:
                        print (file,'-->Good File Name')
                        self.moveGoodFileDir(file)
                    else:
                        print (file,'-->Bad File Name')
                        self.moveBadFileDir(file)
                else:
                    print(file,'-->Bad File Name')
                    self.moveBadFileDir(file)
            else:
                print(file,'-->Bad File Name')
                self.moveBadFileDir(file)
        self.moveBadFilesToArchive()

    def moveGoodFileDir(self,file):
        # '''Copy Good Files to Good_raw dir
        # if dir not available then mkdir'''
        pathDataVal = os.path.join("Data_val/Good_raw/")
        if not os.path.isdir(pathDataVal):
            os.makedirs(pathDataVal)
            shutil.copy(self.batch_directory+file,pathDataVal)
            print('Good Files Copied')
        else:
            shutil.copy(self.batch_directory + file, pathDataVal)
            print('Good Files Copied')

    def moveBadFileDir(self,file):
        # '''Copy Bad Files to Bad_raw dir
        # if dir not available then mkdir'''
        pathDataVal = os.path.join("Data_val/Bad_raw/")
        if not os.path.isdir(pathDataVal):
            os.makedirs(pathDataVal)
            shutil.copy(self.batch_directory+file , pathDataVal)
            print('Bad Files Copied')
        else:
            shutil.copy(self.batch_directory + file, pathDataVal)
            print('Bad Files Copied')


    def moveBadFilesToArchive(self):
        #'''move all bad files to archive & delete bad_raw folder'''
        path = "Data_val/Bad_raw/"
        if os.path.isdir(path):
            for file in listdir(path):
                path_archive = self.makeBadArchiveDir()
                if not os.path.exists(path_archive + file):
                 shutil.move(os.path.join(path+file),os.path.join(path_archive))
            print('All bad files archived')
            shutil.rmtree(path)
            print('Bad Folder dir deleted')
        else: print('No bad files available')

    def makeBadArchiveDir(self):
        path_archive = 'Bad_training_data_Archive/'
        if not os.path.isdir(path_archive):
            os.makedirs(path_archive)
            print('Bad file archive created')
        return path_archive
