from copyreg import pickle
import re
from flask import Flask, request
import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__) # Compulsory to be followed for the app to start at the main/ any other namespace where the app.run() is mentioned.

# Reading the pickle file in read-by mode
data_dir = "./data/"
pickle_in = open(os.path.join(data_dir,"rfclassifier.pkl"),"rb")
rfclassifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = rfclassifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is"+str(prediction)

@app.route('/predict', methods = ["POST"])
def predict_note_afile():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = rfclassifier.predict(df_test)
    return "The predicted value for the csv is"+str(list(prediction))

if __name__ == "__main__":
    app.run()
    #/predict?variance=2&skewness=3&curtosis=2&entropy=1