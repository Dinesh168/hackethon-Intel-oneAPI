from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
modelvect = 'load vectorizer model'
model = "load classification model"
model_path = './Decision.pkl'
recommendation_model = pickle.load(open('Decision.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    title = 'Fake News Detection'
    return render_template('hackthon.html', title=title)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        vect = modelvect.transform(data)
        my_prediction = model.predict(vect)
        output = my_prediction[0]
        if output == 1:
            return render_template('hackthon.html', prediction_text='It is a Spam Message')
        elif output == 0:
            return render_template('hackthon.html', prediction_text='It is not a Spam Message')
