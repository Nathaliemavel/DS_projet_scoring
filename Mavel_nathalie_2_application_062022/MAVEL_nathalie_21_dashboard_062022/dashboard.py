# LIBRAIRIES
from wsgiref.headers import Headers
from pip import main
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
from streamlit_shap import st_shap
import pickle
import lightgbm as lgb
import plotly.express as px
import os
import time
from urllib.request import urlopen
import json
from flask import Flask, render_template, jsonify
import requests


PATH_CHARTS = "./CHARTS/"

################ -FONCTIONS- #######################

# create a charts folder to save the graphs
def create_dir():
  if os.path.isdir(PATH_CHARTS) == False:
    os.makedirs(PATH_CHARTS)

# displays the score on a graphical line
def show_score(point):
    fig, ax = plt.subplots(figsize=(6, 2))
    x = np.arange(0, 1.1, 0.1).tolist()
    y = np.ones((11,), dtype=int)
    plt.plot(x, y, c='black')
    x0 = [2]
    y0 = [1]
    plt.plot(point, 1, "s", c= 'red')
    plt.axvline(x= treshold, c='blue', linestyle='--', label='Valeur limite')
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

################ -LOAD INFORMATIONS- #######################


# Load Dataframe reduce clients
df = pd.read_csv('./DATAS/app_test_reduce_2000.csv')


# Load model with SHAP explanation
with open('./DATAS/model_lgb.pickle', 'rb') as file : 
        LGB = pickle.load(file)

# Load API result model
def prediction(API_URL, data_json):
    data_json = json.dumps(data_json)
    headers = {'Content-Type': 'application/json'}
    response = requests.post(API_URL, data=data_json, headers=headers)

    content = json.loads(response.content.decode('utf-8'))
    y_proba = content['response']

    class_prediction = content['class_prediction']

    if response.status_code != 200:
         return jsonify({
             'status': 'error',
             'message': 'The request to the API did not work. Here is the message returned by the API : {}'.format(content['message'])
         }), 500
    return y_proba, class_prediction


# Logo "Prêt à dépenser"
from PIL import Image
image = Image.open('./logo.png')
st.sidebar.image(image, width=280)

# Loan acceptance threshold
treshold = 1-0.5

################ -APP- #######################
# Title
st.title('DASHBOARD - "Prêt à dépenser"')

# select id client in list clients
list_SK_ID_CURR = df['SK_ID_CURR'].tolist()
list_SK_ID_CURR = list(map(int, list_SK_ID_CURR))
# Selecting applicant ID
select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', list_SK_ID_CURR, key=1)
st.write('You selected: ', select_sk_id)

@st.cache #mise en cache de la fonction pour exécution unique
def data_select_id(id, df):
    return df[df['SK_ID_CURR']==int(id)]

################## -Personnal information client- ##############################

df_client = data_select_id(select_sk_id, df)

# preparation with request API
df_api = df_client.drop(['SK_ID_CURR'], axis=1)
data_json = df_api.to_json(orient='records', lines=True)

index_client = df_client.index
if st.sidebar.checkbox('Show personal data'):
    st.header('PERSONAL INFORMATION')
    st.write("**Age** : {:.0f} ans".format(abs(int(df_client["DAYS_BIRTH_x"]/365))))
    st.write("**Number of children** : {:.0f}".format(df_client["CNT_CHILDREN"].values[0]))
    if df_client["CODE_GENDER_F"].values[0]==1:
        st.write("**Gender** : Female")
    elif df_client["CODE_GENDER_F"].values[0]==0 : 
        st.write("**Gender** : Male")

    st.header('TECHNICAL INFORMATION')
    column_radar_2 = ['EXT_SOURCE_1_x', 'EXT_SOURCE_2_x', 'EXT_SOURCE_3_x']
    radar = df_client[column_radar_2].values[0]
    # fig_radar, ax = plt.subplots(2)
    fig_radar, ax = plt.subplots(figsize=(10, 5))
    fig_radar = px.line_polar(
                    r=radar,
                    theta=column_radar_2,
                    line_close=True,
                    range_r = [0, 1.0],
                    title="External Sources : Values is between 0 and 1")
    st.plotly_chart(fig_radar)

    st.write("**Credit term** : {:.2f}".format(df_client["CREDIT_TERM"].values[0]))
    st.write("**Days employed** : {:.2f}".format(df_client["DAYS_EMPLOYED_PERCENT"].values[0]))
    st.write("**Annuity income** : {:.2f}".format(df_client["ANNUITY_INCOME_PERCENT"].values[0]))
    st.write("**Credit income** : {:.2f}".format(df_client["CREDIT_INCOME_PERCENT"].values[0]))

    st.markdown("**Credit term** : is percentage of the credit amount relative to a client's income.")
    st.markdown("**Days employed** : is percentage of days employed in relation to the client's age.")
    st.markdown("**Annuity income** : is percentage of the loan annuity in relation to the client's income.")
    st.markdown("**Credit income** : is percentage of the loan amount in relation to the client's income.")

###################### -RESULT MODEL- ################################################

if st.sidebar.checkbox('Score and decision'):
    st.header('Score and decision')

    if st.button('run'):
        # Request API scoring (local deploiement)
        # API_URL = "http://127.0.0.1:5000/predict"

        # Request API scoring (heroku deploiement)
        API_URL = "https://apiscoringpretadepenser.herokuapp.com/predict"
        y_proba, class_prediction = prediction(API_URL, data_json)

        # show score and decision
        if class_prediction == 'ACCEPT': 
            st.success('Decision  :  **ACCEPT**')
        else : st.error('Decision  :  **NO ACCEPT**')
        st.write("**SCORE** : {:.2f}".format(y_proba))
        fig = show_score(y_proba)
        st.pyplot(fig=plt)
        st.markdown("**Loan acceptance threshold** is fixed to **0.5**")

################# -EXPLAIN MODEL- #############################################
    
        st.header('Explanation of the rating')
        st.markdown("*Please wait, the explanations may take several minutes to load.*\n")
        with st.spinner('Wait for it...'):
            df = df.drop(['SK_ID_CURR'], axis=1)
            st.markdown("### **GLOBALS** explanations of the model results\n")
            shap.initjs()
            explainer = shap.TreeExplainer(LGB, feature_perturbation="interventional", model_output="raw")
            shap_values = explainer.shap_values(df)
            shap_values = shap.TreeExplainer(LGB).shap_values(df)

            # save and upload
            fig1, ax = plt.subplots(figsize=(50, 50))
            shap.summary_plot(shap_values[1], df)
            create_dir()
            plt.savefig(PATH_CHARTS +'summary_plot.png', format='png', dpi=100, bbox_inches='tight')
            shap_image = Image.open(PATH_CHARTS +'summary_plot.png')
            st.image(shap_image, width=700)
            
            shap.initjs()
            st.markdown("### **INDIVIDUAL** explanation of the model result\n")
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][index_client,:], df.iloc[index_client]))
            
        st.success('success to load explanation for id  : '+str(select_sk_id))
    else:
        pass