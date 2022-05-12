# LIBRAIRIES
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

PATH_CHARTS = "./CHARTS/"

################ -FONCTIONS- #######################

# create a charts folder to save the graphs
def create_dir():
  if os.path.isdir(PATH_CHARTS) == False:
    os.makedirs(PATH_CHARTS)

# displays the score on a graphical line
def show_score():
    fig, ax = plt.subplots(figsize=(6, 2))
    x = np.arange(0, 1.1, 0.1).tolist()
    y = np.ones((11,), dtype=int)

    plt.plot(x, y, c='black')
    x0 = [2]
    y0 = [1]
    plt.plot(LGB_SCORE, 1, "s", c= 'red')
    plt.axvline(x= treshold, c='blue', linestyle='--', label='Valeur limite')
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

################ -LOAD INFORMATIONS- #######################
#Load Dataframe 19 clients
df = pd.read_csv('./DATAS/app_test_reduce_2000.csv')

# Load model
with open('./model_lgb.pickle', 'rb') as file : 
    LGB = pickle.load(file)
    

# Logo "Prêt à dépenser"
from PIL import Image
image = Image.open('./logo.png')
st.sidebar.image(image, width=280)

# Loan acceptance threshold
treshold = 0.5

################ -APP- #######################
# Title
st.title('DASHBOARD - "Prêt à dépenser"')

# select id client in list clients
list_SK_ID_CURR = df['SK_ID_CURR'].tolist()
list_SK_ID_CURR = list(map(int, list_SK_ID_CURR))
# id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
# Selecting applicant ID
select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', list_SK_ID_CURR, key=1)
st.write('You selected: ', select_sk_id)

@st.cache #mise en cache de la fonction pour exécution unique
def data_select_id(id, df):
    return df[df['SK_ID_CURR']==int(id)]

# ################################################
# Personnal information client
df_client = data_select_id(select_sk_id, df)
df_client = df_client.drop(['SK_ID_CURR'], axis=1)
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
    # plt.show()
    st.plotly_chart(fig_radar)

    st.write("**Credit term** : {:.2f}".format(df_client["CREDIT_TERM"].values[0]))
    st.write("**Days employed** : {:.2f}".format(df_client["DAYS_EMPLOYED_PERCENT"].values[0]))
    st.write("**Annuity income** : {:.2f}".format(df_client["ANNUITY_INCOME_PERCENT"].values[0]))
    st.write("**Credit income** : {:.2f}".format(df_client["CREDIT_INCOME_PERCENT"].values[0]))


    st.markdown("**Credit term** : is percentage of the credit amount relative to a client's income.")
    st.markdown("**Days employed** : is percentage of days employed in relation to the client's age.")
    st.markdown("**Annuity income** : is percentage of the loan annuity in relation to the client's income.")
    st.markdown("**Credit income** : is percentage of the loan amount in relation to the client's income.")

# ################################################
# RESULT MODEL                    
if st.sidebar.checkbox('Score and decision'):
    
    st.header('Score and decision')
    LGB_SCORE = (LGB.predict_proba(df_client)[:, 1]).round(3)
    LGB_SCORE = list(map(float, LGB_SCORE))
    st.info('Default in repayment  : '+ str(LGB_SCORE[0]))
    st.write('Default in repayment : Probability that the client will not repay the loan')
    result = pd.DataFrame()
    result['score'] = LGB_SCORE

    for index, row in  result.iterrows():
        if float(row['score']) <= treshold :
            result.at[index,'decision'] = 'ACCEPT'
            result.at[index,'decision_bin'] = 0
        elif float(row['score']) > treshold :
            result.at[index,'decision'] = 'NO ACCEPT'
            result.at[index,'decision_bin'] = 1

    RESULT = result['decision']
    RESULT = list(map(str, RESULT))
    if RESULT[0] =='ACCEPT': 
        st.success('Decision  :  **ACCEPT**')
    else : st.error('Decision  :  **NO ACCEPT**')
        #    st.write('Decision : ', RESULT[0])
    fig = show_score()
    st.pyplot(fig=plt)
    st.markdown("**Loan acceptance threshold** is fixed to **below 0.5**")

# ################################################
# EXPLAIN MODEL  
if st.sidebar.checkbox('Explanation of the rating'):
    st.header('Explanation of the rating')
    st.markdown("*Please wait, the explanations may take several minutes to load.*\n")
    
    with st.spinner('Wait for it...'):
        st.markdown("### **GLOBALS** explanations of the model results\n")
        shap.initjs()
        df = df.drop(['SK_ID_CURR'], axis=1)
        explainer = shap.TreeExplainer(LGB, feature_perturbation="interventional", model_output="raw")
        shap_values = explainer.shap_values(df)
        shap_values = shap.TreeExplainer(LGB).shap_values(df)

        # save and upload
        fig1, ax = plt.subplots(figsize=(50, 30))
        shap.summary_plot(shap_values[1], df)
        # plt.title( "Explanation for SHAP for " + str(select_sk_id))
        create_dir()
        plt.savefig(PATH_CHARTS +'summary_plot.png', format='png', dpi=200, bbox_inches='tight')
        shap_image = Image.open(PATH_CHARTS +'summary_plot.png')
        st.image(shap_image, width=800)
        
        shap.initjs()
        st.markdown("### **INDIVIDUAL** explanation of the model result\n")
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][index_client,:], df.iloc[index_client]))
        # time.sleep()
    st.success('success to load explanation SHAP(SHapley Additive exPlanations) for id  : '+str(select_sk_id))
