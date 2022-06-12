 #!/usr/bin/env python
 # -*-coding:utf-8 -*
import pandas as pd
from flask import Flask, request, render_template
import pickle
import gc
import json


# Instantiate Flask
app = Flask(__name__, template_folder='../templates', static_folder='../static')
#app.config["DEBUG"] = True

# Import model:
model = pickle.load(open("./api_data/model_lgb.pickle", "rb"))

# Set Classification threshold
thresh = 0.50

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    '''If id_client is in the database, it returns y_predict'''
    content_type = request.headers.get('Content-Type')
    if (content_type=='application/json'): 
        data = request.data
        data = data.decode('utf-8')
        data =json.loads(data)
        data = pd.read_json(data, orient='records', lines=True)

        y_proba = model.predict_proba(data)[:, 1]
        y_proba = float(y_proba)
        if y_proba >= thresh:
             class_prediction = "NO ACCEPT"
        else:
             class_prediction = "ACCEPT"
        response = json.dumps({'response': y_proba, 'class_prediction': class_prediction})
        return response, 200

#Local app run
if __name__ == "__main__":
    app.run()