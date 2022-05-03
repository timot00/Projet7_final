# path Desktop\Documents\Data Scientist Openclassrooms\Projet 7

import requests
import json
import datetime
from datetime import date
from sklearn.impute import SimpleImputer
import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import flask
import pickle
#from flask import Flask, render_template, request
import streamlit as st
import shap
from sklearn.cluster import KMeans

@st.cache(allow_output_mutation=True)
def select_col_by_type(df, col_type):
    var_object = []
    for col in df.select_dtypes(col_type):
        var_object.append(col)
    return var_object

@st.cache(allow_output_mutation=True)
def handle_type(df):
    cat_var = select_col_by_type(df, "object")
    int_var = select_col_by_type(df, "int")
    float_var = select_col_by_type(df, "float")
    num_var = int_var + float_var
    return cat_var, num_var

@st.cache(allow_output_mutation=True)
def imputer_nan(df, strategy, var_type):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer = imputer.fit(df[var_type])
    array_type = imputer.transform(df[var_type])
    df_type = pd.DataFrame(array_type, columns=var_type)
    return df_type

@st.cache(allow_output_mutation=True)
def trans_func(df):
    # df = pd.read_csv(data_path +'application_train.csv')
    print('Bonjour 1')
    train_data = df[df.columns[df.isna().sum() / df.shape[0] < 0.5]]
    cat_var, num_var = handle_type(train_data)
    df_cat = imputer_nan(train_data, "most_frequent", cat_var)
    df_num = imputer_nan(train_data, "median", num_var)
    dummies = pd.get_dummies(df_cat, drop_first=True)
    concatenate_df = pd.concat([df_num, dummies], axis=1)
    # data = concatenate_df.sample(n=500, random_state=1)
    #print('Bonjour 2')
    return concatenate_df

@st.cache(allow_output_mutation=True)
def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


columns_list = ['SK_ID_CURR', 'EXT_SOURCE_3', 'DAYS_LAST_PHONE_CHANGE', 'REGION_POPULATION_RELATIVE',
                'TOTALAREA_MODE', 'HOUR_APPR_PROCESS_START', 'DAYS_BIRTH',
                'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'OBS_60_CNT_SOCIAL_CIRCLE',
                'FLAG_DOCUMENT_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2',
                'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_ANNUITY', 'DAYS_ID_PUBLISH',
                'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_GOODS_PRICE',
                # 'NAME_CONTRACT_TYPE', 'CODE_GENDER'
                ]
infos_descrip = ['SK_ID_CURR',
                 "DAYS_BIRTH",
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_CONTRACT_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",
                 "AMT_INCOME_TOTAL"
                 ]

def main() :
    

    @st.cache
    def load_data():
        description = pd.read_csv("features_description.csv", 
                                      usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
        data = pd.read_csv('input_data_model2.zip', index_col='SK_ID_CURR', encoding ='utf-8')
        
        #data = pd.read_csv('input_data_model.csv.zip', index_col='SK_ID_CURR', encoding ='utf-8')
        data = data.drop('Unnamed: 0', 1)
        data = data.drop('index', 1)
        
        
        ML_URI = "https://app-projet7.herokuapp.com/predict"
        

        target = data.iloc[:, -1:]

        #data = data.head(500)
        #target = target.head(500)

        return data, target, description


    def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf

    
    @st.cache(allow_output_mutation=True)
    def load_knn(data):
        knn = knn_training(data)
        return knn
    
    
    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
        round(data["AMT_INCOME_TOTAL"].mean(), 2),
        round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets



    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    
    @st.cache
    def load_age_population(data):
        data_age = -round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    
    @st.cache
    def load_income_population(data):
        df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income
    
    
    @st.cache
    def load_credit_population(data):
        df_credit = pd.DataFrame(data["AMT_CREDIT"])
        df_credit = df_credit.loc[df_credit['AMT_CREDIT'] < 2e6, :]
        return df_credit


    @st.cache
    def load_prediction(data, id, clf):
        X=data.iloc[:, 1:]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score

    
    @st.cache
    def load_kmeans(data, id, mdl):
        index = data[data.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(data.loc[data.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].data(10)

    
    @st.cache
    def knn_training(data):
        knn = KMeans(n_clusters=2).fit(data)
        return knn 



    #Loading data……
    data, target, description = load_data()
    id_client = data.index.values
    clf = load_model()
    
    
    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <div style="padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard de "scoring crédit" évaluation de la solvabilité client</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    
    # Sélection du numéro client (id)
    chk_id = st.sidebar.selectbox("ID Client", id_client)

    # On charge les data client
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)


    # On affiche les infos dans la barre latérale:
    
    # Nombre de clients dans l'échantillon
    st.sidebar.markdown("<u>Nombre de clients dans le jeu de données :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Revenu moyen
    st.sidebar.markdown("<u>Revenu annuel moyen (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(round(rev_moy))

    # Crédit AMT
    st.sidebar.markdown("<u>Montant moyen de l'emprunt (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(round(credits_moy))
    
    # Camembert
    #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.pie(targets, explode=[0, 0.1], labels=['Remboursement OK', 'Défaut de paiment'], autopct='%1.1f%%', startangle=90)
    st.sidebar.pyplot(fig)
    
    
    
    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    # Champs de la barre latérale:
    st.write("ID du client:", chk_id)
    st.header("**Informations du client**")


    infos_client = identite_client(data, chk_id)
    if round(infos_client["CODE_GENDER"].values[0])==0:
        gender="F"
    else:
        gender="M"
    st.write("**Gender : **", gender)
    st.write("**Age : **{:.0f} ans".format(int(-infos_client["DAYS_BIRTH"]/365)))
    #st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))    

    st.write("**Revenu annuel : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write("**Montant du crédit annualisé : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
    #st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))


    st.subheader("*Positionnement du client dans la base de données:*")
    # Graphique de l'age / aux autres clients
    data_age = load_age_population(data)
    fig, ax = plt.subplots(1, 3, figsize=(6,3))
    sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20, ax=ax[0])
    ax[0].axvline(int(-infos_client["DAYS_BIRTH"].values / 365), color="blue", linestyle='--')
    ax[0].set(title='Ages', xlabel='', ylabel='')
    fig.tight_layout()

    # Graphique du revenu / aux autres clients
    data_income = load_income_population(data)
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10, ax=ax[1])
    ax[1].axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="blue", linestyle='--')
    ax[1].set(title='Revenus (USD)', xlabel='', ylabel='')

    # Graphique du montant du crédit / aux autres clients
    data_credit = load_credit_population(data)
    sns.histplot(data_credit["AMT_CREDIT"], edgecolor = 'k', color="goldenrod", bins=10, ax=ax[2])
    ax[2].axvline(int(infos_client["AMT_CREDIT"].values[0]), color="blue", linestyle='--')
    ax[2].set(title='Crédits (USD)', xlabel='', ylabel='')
    st.pyplot(fig)

    
    ## Affichage de la solvabilité client ##
    
    #
    st.header("**Score de solvabilité**")
    prediction = load_prediction(data, chk_id, clf)
    st.write("**Probabilité d'un défaut de paiement : **{:.0f} %".format(round(float(prediction)*100, 2)))


    st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

    
    # Feature importance / description
    st.header("**Feature importance:**")
    shap.initjs()
    X = data.iloc[:, 1:]
    X = X[X.index == chk_id]
    number = st.slider("Choisir un nombre de features:", 0, 20, 5)

    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(load_model())
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
    st.pyplot(fig)


    st.markdown("Description des features:", unsafe_allow_html=True)
    list_features = description.index.to_list()
    feature = st.selectbox('Choisir une feature:', list_features)
    st.table(description.loc[description.index == feature][:1])
 
    
        
if __name__ == '__main__':
    main()

