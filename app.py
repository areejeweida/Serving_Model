from flask import Flask, request, redirect, url_for, flash, jsonify
import pickle
from sklearn.linear_model import LinearRegression
import json
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load the model
model = pickle.load(open('bike_model_LinReg.pkl', 'rb'))


@app.route("/predict_single")
def predict():
    # Retrieve query parameters related to this request.
    temperature = float(request.args.get('temperature'))
    humidity = float(request.args.get('humidity'))
    windspeed = float(request.args.get('windspeed'))
    day = float(request.args.get('day'))
    month = float(request.args.get('month'))
    year = float(request.args.get('year'))

    # Our model expects a list of features
    features = [[temperature, humidity, windspeed, day, month, year]]

    # Use the model to predict
    prediction = model.predict(features)[0]

    # Create and send a response to the API caller
    return jsonify(prediction=prediction)


@app.route('/predict', methods=['GET', 'POST'])
def results():
    # reads the received json
    data = request.json
    res = dict()
    res['temperature'] = dict()
    res['humidity'] = dict()
    res['windspeed'] = dict()
    res['day'] = dict()
    res['month'] = dict()
    res['year'] = dict()
    res['pred'] = dict()

    for key in data.keys():
        sample = data[key]
        res['temperature'][key] = sample['temperature']
        res['humidity'][key] = sample['humidity']
        res['windspeed'][key] = sample['windspeed']
        res['day'][key] = sample['day']
        res['month'][key] = sample['month']
        res['year'][key] = sample['year']

        # Our model expects a list of features
        features = [[sample['temperature'], sample['humidity'],sample['windspeed'],
                     sample['day'], sample['month'], sample['year']]]

        # Use the model to predict
        res['pred'][key] = model.predict(features)[0]

    # returns a json file
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)
