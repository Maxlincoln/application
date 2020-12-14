from flask import Flask, request, json
import flask
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import boto3
import joblib
import pickle

BUCKET_NAME = 'mldistribuido'
MODEL_FILE_NAME = 'model.dat'

S3 = boto3.client('s3', region_name='us-east-1')

def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

@app.route('/predict', methods=['POST'])
def predict():

     data = flask.request.get_json(force=True)['features']
     df = pd.DataFrame(np.array(data), columns = ["shop_id","item_id","item_category_id","month","year"])

     #model = xgb.Booster({'nthread': 4})
     #model.load_model(MODEL_FILE_NAME)

     #model = joblib.load(MODEL_FILE_NAME )
     model = load_model(MODEL_FILE_NAME)

     xgbpredict = xgb.DMatrix(df)
     prediction = model.predict(xgbpredict).tolist()

     result = {'prediction': prediction} #prediction

     return json.dumps(result)

@memoize
def load_model(key):

    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    model_str = response['Body'].read()

    model = pickle.loads(model_str)

    return model

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=port)
